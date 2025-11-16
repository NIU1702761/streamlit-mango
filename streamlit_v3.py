#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ==========================
# CONFIG
# ==========================

V3_PATH    = "submission_v3_balanced.csv"
STACK_PATH = "submission_stacking_lgbm_catboost.csv"
RIDGE_PATH = "submission_ridge_pca.csv"


# ==========================
# FUNCIONS AUXILIARS
# ==========================

@st.cache_data
def load_submissions(v3_path, stack_path, ridge_path):
    """Carrega i fusiona les tres submissions per ID."""
    v3 = pd.read_csv(v3_path).sort_values("ID")
    stack = pd.read_csv(stack_path).sort_values("ID")
    ridge = pd.read_csv(ridge_path).sort_values("ID")

    # Assegurem que tenen els mateixos IDs
    df = pd.DataFrame({"ID": v3["ID"].values})
    df["v3"] = v3["Production"].values
    df["stack"] = stack["Production"].values
    df["ridge"] = ridge["Production"].values

    return df


def compute_ensemble(df, w_v3, w_stack, w_ridge, boost):
    """Calcula l’ensemble amb pesos i boost donats."""
    weights_sum = w_v3 + w_stack + w_ridge
    if weights_sum <= 0:
        weights_sum = 1.0

    base_pred = (
        w_v3   * df["v3"] +
        w_stack * df["stack"] +
        w_ridge * df["ridge"]
    ) / weights_sum

    final_pred = base_pred * boost
    final_pred = np.maximum(final_pred, 0.0)  # Evitar negatius

    df_out = df.copy()
    df_out["ensemble"] = final_pred
    return df_out


def basic_stats(series):
    return {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "median": float(series.median()),
    }


# ==========================
# APLICACIÓ STREAMLIT
# ==========================

def main():
    st.set_page_config(
        page_title="Mango Ensemble Supermodel",
        layout="wide",
    )

    st.title("🧬 Mango Datathon — Ensemble Supermodel (v3 + Stacking + Ridge)")

    st.markdown(
        """
Aquesta demo mostra com combinem **tres models diferents** per obtenir una única
predicció de `Production` per a cada producte:

- `v3`: model robust, amb moltes features → **pes principal**
- `stacking`: model amb LightGBM + CatBoost → **segon pes**
- `ridge`: model més suau basat en PCA → **pes petit estabilitzador**

L’ensemble permet **reduir la variància** i controlar millor el **nivell de producció**
mitjançant un factor de *boost*.
"""
    )

    # 1) Carregar dades
    with st.spinner("Carregant submissions..."):
        df = load_submissions(V3_PATH, STACK_PATH, RIDGE_PATH)

    st.subheader("1️⃣ Dades carregades")
    st.write(f"Nombre de productes (files): **{len(df)}**")
    st.dataframe(df.head())

    # 2) Sidebar per ajustar pesos i boost
    st.sidebar.header("⚙️ Paràmetres de l’ensemble")

    st.sidebar.markdown("### Pesos dels models")
    w_v3 = st.sidebar.slider("Pes v3",    min_value=0.0, max_value=1.0, value=0.55, step=0.05)
    w_stack = st.sidebar.slider("Pes stacking", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    w_ridge = st.sidebar.slider("Pes ridge",    min_value=0.0, max_value=1.0, value=0.10, step=0.05)

    st.sidebar.markdown("### Boost global")
    boost = st.sidebar.slider("Factor de boost", min_value=1.00, max_value=1.30, value=1.08, step=0.01)

    st.sidebar.markdown(
        f"""
**Pesos normalitzats** (suma = 1):

- v3: `{w_v3:.2f}`
- stacking: `{w_stack:.2f}`
- ridge: `{w_ridge:.2f}`

Boost aplicat: `{boost:.2f}x`
"""
    )

    # 3) Calcul ensemble
    df_ens = compute_ensemble(df, w_v3, w_stack, w_ridge, boost)

    st.subheader("2️⃣ Estadístiques bàsiques per model i ensemble")

    col1, col2, col3, col4 = st.columns(4)
    stats_v3 = basic_stats(df_ens["v3"])
    stats_stack = basic_stats(df_ens["stack"])
    stats_ridge = basic_stats(df_ens["ridge"])
    stats_ens = basic_stats(df_ens["ensemble"])

    with col1:
        st.markdown("**Model v3**")
        st.write(stats_v3)
    with col2:
        st.markdown("**Model stacking**")
        st.write(stats_stack)
    with col3:
        st.markdown("**Model ridge**")
        st.write(stats_ridge)
    with col4:
        st.markdown("**Ensemble final**")
        st.write(stats_ens)

    st.caption(
        "Aquestes estadístiques permeten veure si l’ensemble queda massa per sota o per sobre "
        "del rang de produccions que donen els models individuals."
    )

    # 4) Histograma de la predicció final
    st.subheader("3️⃣ Distribució de la predicció de Production (ensemble)")

    hist = (
        alt.Chart(df_ens)
        .mark_bar()
        .encode(
            x=alt.X("ensemble:Q", bin=alt.Bin(maxbins=40), title="Production (ensemble)"),
            y=alt.Y("count():Q", title="Nombre de productes"),
        )
    )
    st.altair_chart(hist, use_container_width=True)

    st.caption(
        "La forma de la distribució mostra com queden repartides les produccions "
        "un cop combinats els models i aplicat el boost."
    )

    # 5) Comparació v3 vs ensemble
    st.subheader("4️⃣ Comparació entre v3 i ensemble")

    scatter_df = df_ens[["v3", "ensemble"]].copy()
    scatter_chart = (
        alt.Chart(scatter_df)
        .mark_circle(size=30, opacity=0.5)
        .encode(
            x=alt.X("v3:Q", title="Predicció v3"),
            y=alt.Y("ensemble:Q", title="Predicció ensemble"),
        )
    )

    st.altair_chart(scatter_chart, use_container_width=True)
    st.caption(
        "Si tots els punts estiguessin a la diagonal, l’ensemble i v3 serien iguals. "
        "Les desviacions mostren on el stacking i el ridge estan modificant la predicció."
    )

    # 6) Explorador d'un producte concret
    st.subheader("5️⃣ Explorador d'un producte")

    selected_id = st.selectbox("Selecciona un ID:", df_ens["ID"].values)

    row = df_ens[df_ens["ID"] == selected_id].iloc[0]

    st.write(f"**ID:** `{selected_id}`")
    st.write(
        f"- Predicció v3: `{row['v3']:.2f}`\n"
        f"- Predicció stacking: `{row['stack']:.2f}`\n"
        f"- Predicció ridge: `{row['ridge']:.2f}`\n"
        f"- **Ensemble final:** `{row['ensemble']:.2f}`"
    )

    st.caption(
        "Això permet explicar, per a un producte concret, com cada model hi contribueix "
        "i com el pes i el boost acaben afectant la producció final."
    )

    # 7) Descarregar CSV de submission
    st.subheader("6️⃣ Descarrega la submission final")

    submission = df_ens[["ID", "ensemble"]].rename(columns={"ensemble": "Production"})
    csv_bytes = submission.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Descarrega `submission_ensemble_supermodel.csv`",
        data=csv_bytes,
        file_name="submission_ensemble_supermodel.csv",
        mime="text/csv",
    )

    st.caption(
        "Aquest fitxer es pot pujar directament al Kaggle. "
        "Si canvies els pesos o el boost, torna a descarregar el CSV actualitzat."
    )


if __name__ == "__main__":
    main()