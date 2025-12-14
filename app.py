import io
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Try to import RDKit – if not installed, show a nice error
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKit_AVAILABLE = True
except ImportError:
    RDKit_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


# ============ UTILITY FUNCTIONS ============

def clean_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and keep only required columns."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Try to map likely column names to standard ones
    col_map = {}
    for col in df.columns:
        low = col.lower()
        if "name" in low and "dock" not in low:
            col_map[col] = "name"
        elif "smiles" in low:
            col_map[col] = "smiles"
        elif ("dock" in low and "score" in low) or low == "score":
            col_map[col] = "docking_score"

    df = df.rename(columns=col_map)

    missing = [c for c in ["smiles", "docking_score"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Ensure docking_score is numeric
    df["docking_score"] = pd.to_numeric(df["docking_score"], errors="coerce")

    # Drop rows with missing values
    df = df.dropna(subset=["smiles", "docking_score"]).reset_index(drop=True)

    # If no explicit name column, create one
    if "name" not in df.columns:
        df["name"] = [f"Mol_{i+1}" for i in range(len(df))]

    # Reorder columns nicely
    df = df[["name", "smiles", "docking_score"] + [c for c in df.columns if c not in ["name", "smiles", "docking_score"]]]

    return df


def featurize_smiles(smiles: str):
    """Convert SMILES → RDKit Morgan fingerprint + simple descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 2048-bit Morgan fingerprint (radius 2)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_arr = np.array(fp)

    # Simple descriptors
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rotb = Descriptors.NumRotatableBonds(mol)

    extra = np.array([mw, logp, tpsa, hbd, hba, rotb])
    return np.concatenate([fp_arr, extra])


def build_features(df: pd.DataFrame):
    """Build feature matrix X and filtered dataframe df_valid."""
    features = []
    valid_idx = []

    for i, s in enumerate(df["smiles"]):
        f = featurize_smiles(s)
        if f is not None:
            features.append(f)
            valid_idx.append(i)

    if len(features) == 0:
        raise ValueError("RDKit could not featurize any SMILES.")

    X = np.vstack(features)
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    y = df_valid["docking_score"].values

    return X, y, df_valid


def train_rf_regressor(X, y):
    """Train a RandomForest regressor and return model + metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="r2")

    metrics = {
        "rmse": rmse,
        "r2": r2,
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std()
    }

    return rf, metrics


def make_scatter_plot(df_valid: pd.DataFrame):
    """Scatter plot of Docking vs ML score with labels."""
    fig, ax = plt.subplots(figsize=(6, 5))

    x = df_valid["docking_score"]
    y = df_valid["ml_pred_score"]

    ax.scatter(x, y)
    for _, row in df_valid.iterrows():
        ax.text(row["docking_score"], row["ml_pred_score"], row["name"],
                fontsize=7)

    ax.set_xlabel("Docking score (kcal/mol)")
    ax.set_ylabel("ML-predicted docking score")
    ax.invert_xaxis()  # more negative = better
    ax.set_title("AI validation of docking for Tau-stabilizing ligands")
    fig.tight_layout()
    return fig


def make_heatmap(df_consensus: pd.DataFrame):
    """Heatmap of docking_score, ml_pred_score, consensus_rank."""
    metrics = ["docking_score", "ml_pred_score", "consensus_rank"]
    heat = df_consensus[metrics].copy()

    # Flip signs so higher = better for visualization
    heat["docking_score"] = -heat["docking_score"]
    heat["ml_pred_score"] = -heat["ml_pred_score"]
    heat["consensus_rank"] = -heat["consensus_rank"]

    # Z-score each column
    heat = (heat - heat.mean()) / heat.std()
    values = heat.values

    fig, ax = plt.subplots(figsize=(5, 6))
    im = ax.imshow(values, aspect="auto")

    ax.set_yticks(range(len(df_consensus)))
    ax.set_yticklabels(df_consensus["name"])
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Z-score (higher = better)")
    ax.set_title("Ligand–metric heatmap")
    fig.tight_layout()
    return fig


def run_pipeline(df_input: pd.DataFrame):
    """End-to-end pipeline returning enriched dataframe + figs + metrics."""
    df = clean_input_df(df_input)

    X, y, df_valid = build_features(df)

    rf_reg, metrics = train_rf_regressor(X, y)

    # Add ML predictions
    df_valid["ml_pred_score"] = rf_reg.predict(X)

    # Consensus ranking
    df_valid["dock_rank"] = df_valid["docking_score"].rank(ascending=True)
    df_valid["ml_rank"] = df_valid["ml_pred_score"].rank(ascending=True)
    df_valid["consensus_rank"] = (
        df_valid["dock_rank"] + df_valid["ml_rank"]
    ) / 2

    df_consensus = df_valid.sort_values("consensus_rank").reset_index(drop=True)

    # Figures
    scatter_fig = make_scatter_plot(df_valid)
    heatmap_fig = make_heatmap(df_consensus)

    return df_consensus, metrics, scatter_fig, heatmap_fig


# ============ STREAMLIT APP ============

def main():
    st.set_page_config(
        page_title="Tau Stabilizer AI Dashboard",
        layout="wide"
    )

    st.title("Tau Stabilizer AI/ML Dashboard")
    st.write(
        "Upload a CSV with columns for ligand **name**, **SMILES**, and "
        "**docking_score**, and this app will:\n"
        "- Clean the data\n"
        "- Featurize SMILES with RDKit\n"
        "- Train a Random Forest model to learn structure–docking relationships\n"
        "- Compute ML-predicted scores and consensus ranking\n"
        "- Generate a scatter plot and heatmap\n"
        "- Let you download the enriched results table"
    )

    if not RDKit_AVAILABLE:
        st.error(
            "RDKit is not installed in this environment. "
            "Please make sure RDKit is available before running this app."
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload your tau docking CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        st.subheader("Preview of uploaded data")
        st.dataframe(df_input.head())

        if st.button("Run AI/ML Pipeline"):
            with st.spinner("Running pipeline... this may take a moment."):
                try:
                    df_out, metrics, scatter_fig, heatmap_fig = run_pipeline(df_input)
                except Exception as e:
                    st.error(f"Error in pipeline: {e}")
                    st.stop()

            st.success("Pipeline completed successfully!")

            # Metrics
            st.subheader("Model Performance (Random Forest Regression)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Test RMSE", f"{metrics['rmse']:.3f}")
            col2.metric("Test R²", f"{metrics['r2']:.3f}")
            col3.metric("5-fold CV R²", f"{metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}")

            # Figures
            st.subheader("AI Validation Plots")
            c1, c2 = st.columns(2)
            with c1:
                st.pyplot(scatter_fig)
            with c2:
                st.pyplot(heatmap_fig)

            # Top hits table
            st.subheader("Consensus Ranking of Ligands")
            st.dataframe(
                df_out[["name", "docking_score", "ml_pred_score",
                        "dock_rank", "ml_rank", "consensus_rank"]]
            )

            # Download enriched CSV
            csv_buffer = io.StringIO()
            df_out.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download enriched results as CSV",
                data=csv_buffer.getvalue(),
                file_name="tau_docking_with_ml.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()