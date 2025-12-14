import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tau Stabilizer Viewer", layout="wide")
st.title("Tau Stabilizer Results Dashboard")

st.write(
    "Upload the CSV you exported from Kaggle (e.g., `tau_docking_with_ml.csv`) "
    "containing at least `name`, `docking_score`, `ml_pred_score`, and "
    "`consensus_rank`. This app will visualize the AI/docking results."
)

uploaded_file = st.file_uploader("Upload enriched results CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())

    required_cols = ["name", "docking_score", "ml_pred_score", "consensus_rank"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # Sort by consensus
    df_consensus = df.sort_values("consensus_rank").reset_index(drop=True)

    st.subheader("Top hits by consensus rank")
    st.dataframe(
        df_consensus[["name", "docking_score", "ml_pred_score", "consensus_rank"]]
        .head(10)
    )

    # Scatter plot
    st.subheader("Docking vs ML-predicted score")
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.scatter(df["docking_score"], df["ml_pred_score"])
    for _, row in df.iterrows():
        ax1.text(row["docking_score"], row["ml_pred_score"], row["name"], fontsize=7)
    ax1.set_xlabel("Docking score (kcal/mol)")
    ax1.set_ylabel("ML-predicted docking score")
    ax1.invert_xaxis()
    ax1.set_title("AI validation of docking")
    fig1.tight_layout()
    st.pyplot(fig1)

    # Heatmap
    st.subheader("Ligand–metric heatmap")
    metrics = ["docking_score", "ml_pred_score", "consensus_rank"]
    heat = df_consensus[metrics].copy()

    # Flip signs so higher = better
    heat["docking_score"] = -heat["docking_score"]
    heat["ml_pred_score"] = -heat["ml_pred_score"]
    heat["consensus_rank"] = -heat["consensus_rank"]

    heat = (heat - heat.mean()) / heat.std()
    values = heat.values

    fig2, ax2 = plt.subplots(figsize=(5, 6))
    im = ax2.imshow(values, aspect="auto")
    ax2.set_yticks(range(len(df_consensus)))
    ax2.set_yticklabels(df_consensus["name"])
    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels(metrics, rotation=45, ha="right")
    cbar = fig2.colorbar(im, ax=ax2)
    cbar.set_label("Z-score (higher = better)")
    ax2.set_title("Ligand–metric heatmap")
    fig2.tight_layout()
    st.pyplot(fig2)

    # Allow download (e.g., if you sorted or filtered)
    csv_buf = io.StringIO()
    df_consensus.to_csv(csv_buf, index=False)
    st.download_button(
        "Download sorted results CSV",
        data=csv_buf.getvalue(),
        file_name="tau_docking_with_ml_sorted.csv",
        mime="text/csv",
    )
