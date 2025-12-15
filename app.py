# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


# ============ DATA CLEANING & FEATURE BUILDING ============

def clean_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names and make sure we have:
    - name
    - smiles
    - docking_score
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

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

    # If name column missing, create generic names
    if "name" not in df.columns:
        df["name"] = [f"Mol_{i+1}" for i in range(len(df))]

    fixed = ["name", "smiles", "docking_score"]
    others = [c for c in df.columns if c not in fixed]
    df = df[fixed + others]

    return df


def featurize_smiles(smiles: str):
    """SMILES → 2048-bit Morgan fingerprint + simple descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_arr = np.array(fp, dtype=np.int8)

    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd  = Descriptors.NumHDonors(mol)
    hba  = Descriptors.NumHAcceptors(mol)
    rotb = Descriptors.NumRotatableBonds(mol)

    extra = np.array([mw, logp, tpsa, hbd, hba, rotb], dtype=float)
    return np.concatenate([fp_arr, extra])


def build_features(df: pd.DataFrame):
    """Build feature matrix X and y (docking_score), plus filtered df."""
    features = []
    valid_idx = []

    for i, s in enumerate(df["smiles"]):
        f = featurize_smiles(s)
        if f is not None:
            features.append(f)
            valid_idx.append(i)

    if not features:
        raise ValueError("RDKit could not featurize any SMILES in this file.")

    X = np.vstack(features)
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    y = df_valid["docking_score"].values

    return X, y, df_valid


# ============ MODEL TRAINING ============

def train_rf_regressor(X, y):
    """Train RandomForestRegressor and return model + metrics."""
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


# ============ PLOTTING HELPERS ============

def make_scatter_plot(df_valid: pd.DataFrame):
    """Docking vs ML scatter with labels."""
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
    ax.set_title("AI validation of docking")
    fig.tight_layout()
    return fig


def make_heatmap(df_consensus: pd.DataFrame):
    """Heatmap of docking_score, ml_pred_score, consensus_rank."""
    metrics = ["docking_score", "ml_pred_score", "consensus_rank"]
    heat = df_consensus[metrics].copy()

    # Flip sign so higher = better for display
    heat["docking_score"] = -heat["docking_score"]
    heat["ml_pred_score"] = -heat["ml_pred_score"]
    heat["consensus_rank"] = -heat["consensus_rank"]

    # Z-score
    heat = (heat - heat.mean()) / heat.std()
    vals = heat.values

    fig, ax = plt.subplots(figsize=(5, 6))
    im = ax.imshow(vals, aspect="auto")
    ax.set_yticks(range(len(df_consensus)))
    ax.set_yticklabels(df_consensus["name"])
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Z-score (higher = better)")
    ax.set_title("Ligand–metric heatmap")
    fig.tight_layout()
    return fig


# ============ END-TO-END PIPELINE ============

def run_pipeline(df_input: pd.DataFrame):
    """
    End-to-end:
    - clean data
    - featurize
    - train RF
    - add ml_pred_score
    - build consensus ranks
    - return enriched df + metrics + figs
    """
    df = clean_input_df(df_input)
    X, y, df_valid = build_features(df)

    rf_model, metrics = train_rf_regressor(X, y)

    # ML predictions for all valid molecules
    df_valid["ml_pred_score"] = rf_model.predict(X)

    # Rankings (lower docking_score = better)
    df_valid["dock_rank"] = df_valid["docking_score"].rank(ascending=True)
    df_valid["ml_rank"] = df_valid["ml_pred_score"].rank(ascending=True)
    df_valid["consensus_rank"] = (
        df_valid["dock_rank"] + df_valid["ml_rank"]
    ) / 2.0

    df_consensus = df_valid.sort_values("consensus_rank").reset_index(drop=True)

    scatter_fig = make_scatter_plot(df_valid)
    heatmap_fig = make_heatmap(df_consensus)

    return df_consensus, metrics, scatter_fig, heatmap_fig


# ============ STREAMLIT APP ============

def main():
    st.set_page_config(
        page_title="Tau Stabilizer AI Pipeline",
        layout="wide"
    )

    st.title("Tau Stabilizer AI/ML Pipeline")
    st.write(
        "Upload a docking results file (CSV or Excel) containing ligand **name**, "
        "**SMILES**, and **docking score**. This app will:\n"
        "1. Clean and standardize the data\n"
        "2. Featurize SMILES with RDKit (Morgan fingerprints + descriptors)\n"
        "3. Train a Random Forest model to learn structure–docking relationships\n"
        "4. Compute ML-predicted scores and a consensus rank\n"
        "5. Visualize the results and let you download `tau_docking_with_ml.csv`"
    )

    uploaded_file = st.file_uploader(
        "Upload docking results file (.csv or .xlsx)",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        # --- Read file depending on extension ---
        file_name = uploaded_file.name.lower()
        try:
            if file_name.endswith(".csv"):
                df_input = pd.read_csv(uploaded_file)
            else:
                df_input = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        st.subheader("Preview of uploaded data")
        st.dataframe(df_input.head())

        if st.button("Run AI/ML pipeline"):
            with st.spinner("Running pipeline..."):
                try:
                    df_out, metrics, scatter_fig, heatmap_fig = run_pipeline(df_input)
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.stop()

            st.success("Pipeline completed successfully!")

            # --- Model performance ---
            st.subheader("Random Forest Regression Performance")
            c1, c2, c3 = st.columns(3)
            c1.metric("Test RMSE", f"{metrics['rmse']:.3f}")
            c2.metric("Test R²", f"{metrics['r2']:.3f}")
            c3.metric(
                "5-fold CV R²",
                f"{metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}"
            )

            # --- Plots ---
            st.subheader("AI Validation Plots")
            pc1, pc2 = st.columns(2)
            with pc1:
                st.pyplot(scatter_fig)
            with pc2:
                st.pyplot(heatmap_fig)

            # --- Top hits table ---
            st.subheader("Consensus-ranked ligands")
            st.dataframe(
                df_out[[
                    "name", "smiles",
                    "docking_score", "ml_pred_score",
                    "dock_rank", "ml_rank", "consensus_rank"
                ]]
            )

            # --- Download enriched CSV as tau_docking_with_ml.csv ---
            csv_buf = io.StringIO()
            df_out.to_csv(csv_buf, index=False)
            st.download_button(
                label="Download enriched results (tau_docking_with_ml.csv)",
                data=csv_buf.getvalue(),
                file_name="tau_docking_with_ml.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
