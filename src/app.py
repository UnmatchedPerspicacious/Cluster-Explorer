"""
app.py
------
Streamlit dashboard for the Cluster Explorer.

Upload any CSV, pick features, choose a clustering algorithm,
and get an interactive PCA + t-SNE scatter plot in seconds.

Run:
    streamlit run src/app.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from features import run_pipeline, elbow_scores

SAMPLE_DIR = Path(__file__).parent.parent / "sample_data"

SAMPLES = {
    "Iris (150 rows, 4 features)":           SAMPLE_DIR / "iris.csv",
    "Wine (178 rows, 13 features)":          SAMPLE_DIR / "wine.csv",
    "Breast Cancer (569 rows, 30 features)": SAMPLE_DIR / "breast_cancer.csv",
}

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cluster Explorer",
    page_icon="🔭",
    layout="wide",
)

# ── Theme ──────────────────────────────────────────────────────────────────
BG     = "#0f0e17"
CARD   = "#1a1928"
ACCENT = "#e8b86d"
PURPLE = "#9b8ec4"
GREEN  = "#5fcf80"
RED    = "#e05c5c"
MUTED  = "#7a78a0"
TEXT   = "#e8e6f0"

CLUSTER_COLORS = [
    "#e8b86d", "#9b8ec4", "#5fcf80", "#e05c5c",
    "#5bc8f5", "#f5a623", "#bd10e0", "#7ed321",
    "#d0021b", "#4a90e2",
]

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   CARD,
    "axes.edgecolor":   MUTED,
    "axes.labelcolor":  MUTED,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "text.color":       TEXT,
    "grid.color":       "#2d2b45",
    "grid.alpha":       0.4,
})

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; color: {TEXT}; }}
    .block-container {{ padding-top: 2rem; }}
    .metric-card {{
        background: {CARD};
        border: 1px solid #2d2b45;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
    }}
    .metric-value {{ font-size: 1.8rem; font-weight: 800; color: {ACCENT}; }}
    .metric-label {{ font-size: 0.8rem; color: {MUTED}; margin-top: 4px; letter-spacing: 0.06em; }}
    .info-box {{
        background: {CARD};
        border: 1px solid #2d2b45;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
        font-size: 0.9rem;
        color: {MUTED};
    }}
    h1, h2, h3 {{ color: {TEXT} !important; }}
    div[data-testid="stSidebar"] {{ background-color: {CARD}; }}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Data")

    data_source = st.radio(
        "Data source",
        ["Upload your own CSV", "Use a sample dataset"],
        label_visibility="collapsed",
    )

    df = None

    if data_source == "Upload your own CSV":
        st.caption("⚠️ Maximum 2,000 rows — t-SNE is computationally expensive on CPU.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            if len(df) > 2000:
                st.error(f"Your dataset has {len(df):,} rows — the maximum is 2,000. Please upload a smaller dataset or sample your data down to 2,000 rows before uploading.")
                df = None
            else:
                st.success(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    else:
        sample_name = st.selectbox("Sample dataset", list(SAMPLES.keys()))
        df = pd.read_csv(SAMPLES[sample_name])
        st.info(f"{len(df):,} rows · {len(df.columns)} columns")

    if df is not None:
        st.markdown("---")
        st.markdown("### ⚙️ Features")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("No numeric columns found. Please upload a CSV with numeric features.")
            st.stop()

        feature_cols = st.multiselect(
            "Select feature columns",
            options=numeric_cols,
            default=numeric_cols[:min(len(numeric_cols), 6)],
            help="Only numeric columns can be used as features.",
        )

        color_col = st.selectbox(
            "Colour points by (optional)",
            options=["cluster"] + df.columns.tolist(),
            index=0,
            help="Colour the scatter plot by cluster label or any column in your data.",
        )

        st.markdown("---")
        st.markdown("### 🔬 Algorithm")

        algorithm = st.radio("Clustering algorithm", ["K-Means", "DBSCAN"])

        if algorithm == "K-Means":
            k = st.slider("Number of clusters (k)", min_value=2, max_value=12, value=3)
            show_elbow = st.checkbox("Show elbow curve", value=True)
            eps = min_samples = None
        else:
            eps         = st.slider("Epsilon (neighbourhood size)", 0.1, 5.0, 0.5, 0.1)
            min_samples = st.slider("Min samples per cluster", 2, 20, 5)
            k = show_elbow = None

        st.markdown("---")
        st.markdown("### 🔭 t-SNE")
        perplexity = st.slider(
            "Perplexity",
            min_value=5, max_value=min(50, len(df) - 1), value=30,
            help="Roughly the number of nearest neighbours t-SNE considers. "
                 "Lower = local structure, Higher = global structure."
        )

        run_btn = st.button("Run clustering →", type="primary",
                            disabled=len(feature_cols) < 2)
        if len(feature_cols) < 2:
            st.caption("Select at least 2 feature columns to run.")


# ── Main panel ─────────────────────────────────────────────────────────────
st.title("🔭 Cluster Explorer")
st.markdown(
    "Upload any CSV and explore its natural cluster structure using "
    "**PCA + t-SNE** dimensionality reduction and your choice of "
    "**K-Means** or **DBSCAN** clustering."
)

if df is None:
    st.markdown(f"""
    <div class="info-box">
        👈 Upload a CSV or choose a sample dataset from the sidebar to get started.
        <br><br>
        <b>How it works:</b><br>
        1. Select numeric feature columns<br>
        2. Choose K-Means or DBSCAN<br>
        3. Click Run — PCA compresses the features, t-SNE maps them to 2D,
           and clusters are coloured on the scatter plot
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if "result" not in st.session_state:
    st.session_state.result = None

if run_btn:
    if len(feature_cols) < 2:
        st.error("Please select at least 2 feature columns.")
    else:
        with st.spinner("Running PCA → t-SNE → clustering… this may take 1-3 minutes on CPU"):
            try:
                result = run_pipeline(
                    df, feature_cols, algorithm,
                    perplexity=perplexity,
                    k=k or 3,
                    eps=eps or 0.5,
                    min_samples=min_samples or 5,
                )
                st.session_state.result = result
                st.session_state.color_col  = color_col
                st.session_state.algorithm  = algorithm
                st.session_state.show_elbow = show_elbow
                st.session_state.k          = k
                st.session_state.feature_cols = feature_cols
            except Exception as e:
                st.error(f"Error: {e}")

if st.session_state.result is not None:
    result    = st.session_state.result
    algorithm = st.session_state.algorithm
    n_clusters = len(set(result["cluster"])) - (1 if -1 in result["cluster"].values else 0)
    n_noise    = (result["cluster"] == -1).sum()

    # ── Metrics ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, f"{len(result):,}", "Points"),
        (c2, str(n_clusters), "Clusters found"),
        (c3, str(len(st.session_state.feature_cols)), "Features used"),
        (c4, f"{n_noise:,}" if algorithm == "DBSCAN" else "—", "Noise points"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Main scatter plot ──────────────────────────────────────────────────
    col_scatter, col_right = st.columns([3, 1])

    with col_scatter:
        st.markdown("#### t-SNE Cluster Map")

        color_col = st.session_state.color_col

        fig, ax = plt.subplots(figsize=(10, 7))

        if color_col == "cluster" or color_col not in result.columns:
            # Colour by cluster label
            unique_clusters = sorted(result["cluster"].unique())
            for i, c in enumerate(unique_clusters):
                mask   = result["cluster"] == c
                color  = "#888888" if c == -1 else CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                label  = "Noise" if c == -1 else f"Cluster {c}"
                ax.scatter(result.loc[mask, "x"], result.loc[mask, "y"],
                           c=color, label=label, alpha=0.7, s=20, linewidths=0)
        else:
            # Colour by another column
            col_data = df.loc[result.index, color_col] if color_col in df.columns else result["cluster"]
            if col_data.dtype == object:
                unique_vals = col_data.unique()
                for i, val in enumerate(unique_vals):
                    mask = col_data == val
                    ax.scatter(result.loc[mask, "x"], result.loc[mask, "y"],
                               c=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                               label=str(val), alpha=0.7, s=20, linewidths=0)
            else:
                sc = ax.scatter(result["x"], result["y"],
                                c=col_data, cmap="plasma",
                                alpha=0.7, s=20, linewidths=0)
                plt.colorbar(sc, ax=ax, label=color_col)

        ax.set_xlabel("t-SNE dimension 1")
        ax.set_ylabel("t-SNE dimension 2")
        ax.set_title(f"{algorithm} clustering — {n_clusters} clusters",
                     fontweight="bold", color=TEXT)
        ax.legend(frameon=False, labelcolor=TEXT, fontsize=9,
                  markerscale=2, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown("#### Cluster sizes")
        cluster_counts = result["cluster"].value_counts().sort_index()
        fig2, ax2 = plt.subplots(figsize=(3.5, 4))
        colors = [
            "#888888" if c == -1 else CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            for i, c in enumerate(cluster_counts.index)
        ]
        labels = ["Noise" if c == -1 else f"Cluster {c}" for c in cluster_counts.index]
        ax2.barh(labels, cluster_counts.values, color=colors, height=0.6)
        ax2.set_xlabel("Count")
        ax2.grid(axis="x")
        for spine in ax2.spines.values():
            spine.set_alpha(0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.markdown("#### Stats")
        for c in sorted(result["cluster"].unique()):
            mask  = result["cluster"] == c
            label = "Noise" if c == -1 else f"Cluster {c}"
            pct   = mask.sum() / len(result) * 100
            st.markdown(f"**{label}:** {mask.sum():,} points ({pct:.1f}%)")

    # ── Elbow curve (K-Means only) ─────────────────────────────────────────
    if algorithm == "K-Means" and st.session_state.show_elbow:
        st.markdown("---")
        st.markdown("#### Elbow Curve — choosing the right k")

        with st.spinner("Computing elbow curve…"):
            from features import preprocess, run_pca
            X_scaled, valid_idx = preprocess(df, st.session_state.feature_cols)
            X_pca = run_pca(X_scaled)
            k_range = range(2, min(12, len(result)))
            inertias = elbow_scores(X_pca, k_range)

        fig3, ax3 = plt.subplots(figsize=(8, 3.5))
        ax3.plot(list(k_range), inertias, "o-", color=ACCENT, lw=2, markersize=6)
        ax3.axvline(st.session_state.k, color=PURPLE, lw=1.5, linestyle="--",
                    label=f"Current k = {st.session_state.k}")
        ax3.set_xlabel("Number of clusters (k)")
        ax3.set_ylabel("Inertia")
        ax3.set_xticks(list(k_range))
        ax3.legend(frameon=False, labelcolor=TEXT, fontsize=9)
        ax3.grid(True)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
        st.caption("The elbow point — where inertia stops dropping sharply — is the optimal k.")

    # ── Download results ───────────────────────────────────────────────────
    st.markdown("---")
    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download results as CSV",
        data=csv,
        file_name="cluster_results.csv",
        mime="text/csv",
    )

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='color:{MUTED}; font-size:0.8rem; text-align:center;'>"
    "Pipeline: StandardScaler → PCA → t-SNE → K-Means / DBSCAN · "
    "Built with scikit-learn and Streamlit"
    "</div>",
    unsafe_allow_html=True,
)