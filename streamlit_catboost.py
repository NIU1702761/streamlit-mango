#!/usr/bin/env python
# -*- coding: utf-8 -*-
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ==========================
# CONFIG
# ==========================

# 👉 Canvia aquests paths si cal
TRAIN_PATH = "/Users/aliciamartilopez/Desktop/DATATHON/datathon-fme-2025-mango/train.csv"
TEST_PATH  = "/Users/aliciamartilopez/Desktop/DATATHON/datathon-fme-2025-mango/test.csv"

RANDOM_STATE   = 42
TEST_SIZE      = 0.2
QUANTILE_ALPHA = 0.8


# ==========================
# FUNCIONS DEL MODEL ORIGINAL
# (adaptades per Streamlit)
# ==========================

def preprocess_common(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Preprocessat comú per a train i test:
      - Converteix phase_in / phase_out a dates
      - Crea versions numèriques (dies des de la data mínima)
      - Elimina columnes que no volem com a features
    """

    df = df.copy()

    # Convertim dates
    for col in ["phase_in", "phase_out"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    # Per obtenir una representació numèrica (dies des de la mínima data)
    min_phase_in  = df["phase_in"].min()
    min_phase_out = df["phase_out"].min()

    df["phase_in_days"]  = (df["phase_in"]  - min_phase_in).dt.days
    df["phase_out_days"] = (df["phase_out"] - min_phase_out).dt.days

    # Eliminem les dates originals (string) perquè serien categòriques molt rares
    df.drop(columns=["phase_in", "phase_out"], inplace=True)

    # Podem eliminar també informació purament "weekly", que és sorollosa
    for col in ["weekly_sales", "weekly_demand"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df


def get_feature_sets(df: pd.DataFrame):
    """
    Separa:
      - l'ID
      - el target (Production), si existeix
      - les features numèriques i categòriques
    Retorna:
      X_num, X_cat, y, ids
    """

    df = df.copy()

    # Guardem ID
    ids = df["ID"].copy()

    # Target (només a train)
    y = None
    if "Production" in df.columns:
        y = df["Production"].astype(float)
        df = df.drop(columns=["Production"])

    # Eliminem ID de les features
    df = df.drop(columns=["ID"])

    # Separa tipus
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    X_num = df[numeric_cols].copy()
    X_cat = df[categorical_cols].copy()

    # CatBoost no vol NaN a categòriques
    X_cat = X_cat.fillna("missing").astype(str)

    return X_num, X_cat, y, ids


def compute_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def compute_classification_metrics(y_true_bin, y_pred_bin):
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


# ==========================
# CÀRREGA I PREPROCESSAT (CACHE)
# ==========================

@st.cache_data
def load_and_prepare_data(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path, sep=";")
    test_df  = pd.read_csv(test_path,  sep=";")

    train_df = preprocess_common(train_df, is_train=True)
    test_df  = preprocess_common(test_df,  is_train=False)

    X_num_full, X_cat_full, y_full, train_ids = get_feature_sets(train_df)
    X_num_test, X_cat_test, _, test_ids = get_feature_sets(test_df)

    # Alinear columnes numèriques
    common_num_cols = sorted(list(set(X_num_full.columns) & set(X_num_test.columns)))
    X_num_full = X_num_full[common_num_cols]
    X_num_test = X_num_test[common_num_cols]

    # Alinear columnes categòriques
    common_cat_cols = sorted(list(set(X_cat_full.columns) & set(X_cat_test.columns)))
    X_cat_full = X_cat_full[common_cat_cols]
    X_cat_test = X_cat_test[common_cat_cols]

    return (
        train_df,
        test_df,
        X_num_full,
        X_cat_full,
        y_full,
        train_ids,
        X_num_test,
        X_cat_test,
        test_ids,
    )


# ==========================
# ENTRENAMENT DEL STACKING (CACHE)
# ==========================

@st.cache_resource
def train_stacking_model(X_num_full, X_cat_full, y_full):
    # Train/valid split intern
    idx_train, idx_valid = train_test_split(
        np.arange(len(y_full)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    X_num_train = X_num_full.iloc[idx_train]
    X_num_valid = X_num_full.iloc[idx_valid]
    X_cat_train = X_cat_full.iloc[idx_train]
    X_cat_valid = X_cat_full.iloc[idx_valid]

    y_train = y_full.iloc[idx_train]
    y_valid = y_full.iloc[idx_valid]

    # ----- Model base 1: LightGBM numèric
    lgb_num = LGBMRegressor(
        objective="quantile",
        alpha=QUANTILE_ALPHA,
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
    )
    lgb_num.fit(X_num_train, y_train)

    # ----- Model base 2: CatBoost categòric
    cat_features_idx = list(range(X_cat_train.shape[1]))
    cat_model = CatBoostRegressor(
        depth=6,
        learning_rate=0.05,
        n_estimators=500,
        loss_function=f"Quantile:alpha={QUANTILE_ALPHA}",
        random_state=RANDOM_STATE,
        verbose=False,
    )
    cat_model.fit(
        X_cat_train,
        y_train,
        cat_features=cat_features_idx,
    )

    # ----- Construir features del meta-model
    train_pred_num = lgb_num.predict(X_num_train)
    valid_pred_num = lgb_num.predict(X_num_valid)

    train_pred_cat = cat_model.predict(X_cat_train)
    valid_pred_cat = cat_model.predict(X_cat_valid)

    X_meta_train = pd.DataFrame({
        "pred_num": train_pred_num,
        "pred_cat": train_pred_cat,
    })
    X_meta_valid = pd.DataFrame({
        "pred_num": valid_pred_num,
        "pred_cat": valid_pred_cat,
    })

    # ----- Meta-model LightGBM
    meta_model = LGBMRegressor(
        objective="quantile",
        alpha=QUANTILE_ALPHA,
        n_estimators=300,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )
    meta_model.fit(X_meta_train, y_train)

    final_train_pred = meta_model.predict(X_meta_train)
    final_valid_pred = meta_model.predict(X_meta_valid)

    # Mètriques regressió
    reg_train = compute_regression_metrics(y_train, final_train_pred)
    reg_valid = compute_regression_metrics(y_valid, final_valid_pred)

    # Mètriques classificació (alta/baxa)
    threshold = np.median(y_full)
    y_valid_bin = (y_valid >= threshold).astype(int)
    final_valid_bin = (final_valid_pred >= threshold).astype(int)
    class_valid = compute_classification_metrics(y_valid_bin, final_valid_bin)

    # -------- PART 2: reentrenar amb tot el train per tenir models finals

    # Models base amb TOT el train
    lgb_num_full = LGBMRegressor(
        objective="quantile",
        alpha=QUANTILE_ALPHA,
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
    )
    lgb_num_full.fit(X_num_full, y_full)

    cat_features_full_idx = list(range(X_cat_full.shape[1]))
    cat_model_full = CatBoostRegressor(
        depth=6,
        learning_rate=0.05,
        n_estimators=500,
        loss_function=f"Quantile:alpha={QUANTILE_ALPHA}",
        random_state=RANDOM_STATE,
        verbose=False,
    )
    cat_model_full.fit(
        X_cat_full,
        y_full,
        cat_features=cat_features_full_idx,
    )

    full_pred_num = lgb_num_full.predict(X_num_full)
    full_pred_cat = cat_model_full.predict(X_cat_full)
    X_meta_full = pd.DataFrame({
        "pred_num": full_pred_num,
        "pred_cat": full_pred_cat,
    })

    meta_model_full = LGBMRegressor(
        objective="quantile",
        alpha=QUANTILE_ALPHA,
        n_estimators=300,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
    )
    meta_model_full.fit(X_meta_full, y_full)

    # Calibració P85
    train_pred_full = meta_model_full.predict(X_meta_full)
    p85_true = np.percentile(y_full, 85)
    p85_pred = np.percentile(train_pred_full, 85)
    scale_factor = p85_true / (p85_pred + 1e-6)

    results = {
        "idx_train": idx_train,
        "idx_valid": idx_valid,
        "y_train": y_train,
        "y_valid": y_valid,
        "final_train_pred": final_train_pred,
        "final_valid_pred": final_valid_pred,
        "reg_train": reg_train,
        "reg_valid": reg_valid,
        "class_valid": class_valid,
        "threshold": threshold,
        "lgb_num": lgb_num,
        "cat_model": cat_model,
        "meta_model": meta_model,
        "lgb_num_full": lgb_num_full,
        "cat_model_full": cat_model_full,
        "meta_model_full": meta_model_full,
        "scale_factor": scale_factor,
        "train_pred_full": train_pred_full,
    }

    return results


# ==========================
# APLICACIÓ STREAMLIT
# ==========================

def main():
    st.set_page_config(
        page_title="Mango Datathon - Stacking LGBM + CatBoost",
        layout="wide",
    )

    st.title("🧵 Mango Datathon — Demo Stacking LightGBM + CatBoost")
    st.markdown(
        """
Aquesta demo mostra el **model final** que feu servir:
- LightGBM per a **features numèriques**
- CatBoost per a **features categòriques**
- Un **meta-model LightGBM** que combina les dues prediccions

També fem un **train/validation intern** per veure mètriques reals.
"""
    )

    with st.expander("⚙️ Paràmetres (fixats al codi)", expanded=False):
        st.write(f"Train path: `{TRAIN_PATH}`")
        st.write(f"Test path : `{TEST_PATH}`")
        st.write(f"Random state: `{RANDOM_STATE}`")
        st.write(f"Test size (validació interna): `{TEST_SIZE}`")
        st.write(f"Quantile alpha: `{QUANTILE_ALPHA}`")

    # 1) Carregar i preparar dades
    with st.spinner("Carregant i preprocessant dades..."):
        (
            train_df,
            test_df,
            X_num_full,
            X_cat_full,
            y_full,
            train_ids,
            X_num_test,
            X_cat_test,
            test_ids,
        ) = load_and_prepare_data(TRAIN_PATH, TEST_PATH)

    st.subheader("1️⃣ Dades")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Train**", train_df.shape)
        st.dataframe(train_df.head())
    with col2:
        st.write("**Test**", test_df.shape)
        st.dataframe(test_df.head())

    # 2) Entrenar stacking
    with st.spinner("Entrenant models (stacking)..."):
        model_info = train_stacking_model(X_num_full, X_cat_full, y_full)

    st.subheader("2️⃣ Mètriques de regressió (validació interna)")

    reg_train = model_info["reg_train"]
    reg_valid = model_info["reg_valid"]

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE valid", f"{reg_valid['mae']:.4f}")
    c2.metric("RMSE valid", f"{reg_valid['rmse']:.4f}")
    c3.metric("R² valid", f"{reg_valid['r2']:.4f}")

    st.caption("Mètriques del **meta-model LightGBM** sobre el conjunt de validació interna.")

    # 4) Gràfiques de comportament
    st.subheader("4️3️⃣ Gràfica Real vs Predicció (validació interna)")

    y_valid = model_info["y_valid"]
    final_valid_pred = model_info["final_valid_pred"]

    plot_df = pd.DataFrame({
        "Real": y_valid.values,
        "Pred": final_valid_pred,
    })

    st.scatter_chart(plot_df, x="Real", y="Pred")
    st.caption("Idealment els punts haurien de caure a prop de la diagonal Real=Pred.")

    # 4b) Distribució dels errors
    st.subheader("📊 Distribució dels errors (Pred - Real)")

    # Errors (predicció - valor real)
    residuals = final_valid_pred - y_valid.values
    err_df = pd.DataFrame({"error": residuals})

    # Histograma amb Altair
    hist_chart = (
        alt.Chart(err_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "error:Q",
                bin=alt.Bin(maxbins=30),
                title="Error (Pred - Real)",
            ),
            y=alt.Y(
                "count():Q",
                title="Nombre d'observacions",
            ),
        )
    )

    # Línia vertical a error = 0 (on el model encerta)
    zero_line = alt.Chart(pd.DataFrame({"error": [0]})).mark_rule(
        color="red"
    ).encode(
        x="error:Q"
    )

    # Línia vertical a la mediana de l'error
    median_err = float(np.median(residuals))
    median_line = alt.Chart(pd.DataFrame({"error": [median_err]})).mark_rule(
        color="orange",
        strokeDash=[4, 4],
    ).encode(
        x="error:Q"
    )

    st.altair_chart(hist_chart + zero_line + median_line, use_container_width=True)

    st.caption(
        "La gràfica mostra la distribució dels errors Pred - Real. "
        "La línia vermella és l'error 0 (predicció perfecta) i la línia "
        "taronja és la mediana de l'error: si està per sobre de 0, "
        "el model tendeix a sobreestimar; si està per sota, a infraestimar."
    )

    # 5) Explorador d'un producte (conjunt de validació)
    st.subheader("5️4️⃣ Explorador d'un producte (conjunt de validació)")

    idx_valid = model_info["idx_valid"]
    valid_ids = train_ids.iloc[idx_valid].values

    selected_id = st.selectbox("Selecciona un ID:", valid_ids)

    # localitzar la fila corresponent al validation set
    pos = np.where(valid_ids == selected_id)[0][0]
    real_value = y_valid.iloc[pos]
    pred_value = final_valid_pred[pos]

    st.write(f"**ID**: `{selected_id}`")
    st.write(f"**Production real (train)**: `{real_value:.4f}`")
    st.write(f"**Predicció meta-model**: `{pred_value:.4f}`")
    st.write(f"**Error (pred - real)**: `{(pred_value - real_value):.4f}`")

    st.markdown("**Features originals d'aquest producte:**")
    row = train_df[train_df["ID"] == selected_id].iloc[0]
    st.json(row.to_dict())


if __name__ == "__main__":
    main()