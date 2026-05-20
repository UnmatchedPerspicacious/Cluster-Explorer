# 🔭 Cluster Explorer

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Paradigm](https://img.shields.io/badge/ML-Unsupervised-purple)
![Algorithms](https://img.shields.io/badge/Algorithms-KMeans%20%7C%20DBSCAN-orange)

A universal unsupervised learning tool that lets you upload **any CSV dataset**
and instantly explore its natural cluster structure using **PCA + t-SNE**
dimensionality reduction and your choice of **K-Means** or **DBSCAN** clustering.

---

## Live demo

**[cluster-explorer.streamlit.app](https://cluster-explorer.streamlit.app/)**

---

## What it does

1. Upload any CSV (or use one of the bundled sample datasets)
2. Select which numeric columns to use as features
3. Choose K-Means or DBSCAN and set parameters
4. Click Run — the pipeline runs in seconds:
   - **StandardScaler** normalises all features
   - **PCA** compresses to 50 dimensions (speeds up t-SNE)
   - **t-SNE** maps to 2D for visualisation
   - **K-Means / DBSCAN** assigns cluster labels
5. Interactive scatter plot coloured by cluster, plus an elbow curve for K-Means
6. Download the results as CSV

---

## Algorithms

### K-Means
Partitions data into k clusters by minimising within-cluster variance.
Best for roughly spherical, similarly-sized clusters. The elbow curve
helps you choose the right k.

### DBSCAN
Density-based clustering — finds clusters of arbitrary shape and flags
outliers as noise (label = -1). Best when clusters have irregular shapes
or you don't know k in advance.

---

## Bundled sample datasets

| Dataset | Rows | Features | Source |
|---|---|---|---|
| Iris | 150 | 4 | sklearn |
| Wine | 178 | 13 | sklearn |
| Breast Cancer | 569 | 30 | sklearn |

---

## Run order

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/cluster-explorer.git
cd cluster-explorer

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch the app

```bash
streamlit run src/app.py
```

Opens at `http://localhost:8501`. No training step needed — clustering
runs live in the app when you click Run.

---

## Project structure

```
cluster-explorer/
├── sample_data/
│   ├── iris.csv
│   ├── wine.csv
│   └── breast_cancer.csv
├── src/
│   ├── features.py     ← PCA, t-SNE, K-Means, DBSCAN pipeline
│   └── app.py          ← Streamlit dashboard
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

---

## License

MIT — see [LICENSE](LICENSE).