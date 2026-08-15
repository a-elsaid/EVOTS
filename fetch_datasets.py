"""Fetch Iris, Wine, Breast Cancer (sklearn) and Seeds (UCI) as clean CSVs.
Run from the EVOTS repo root: python fetch_datasets.py
Requires: pip install scikit-learn (if not already installed)
"""
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

OUT = "data/tabular"

def save_sklearn(loader, name):
    d = loader()
    df = pd.DataFrame(d.data, columns=[c.replace(" ", "_") for c in d.feature_names])
    df["target"] = d.target
    path = f"{OUT}/{name}.csv"
    df.to_csv(path, index=False)
    print(f"{name}: {df.shape[0]} rows, {df.shape[1]-1} features, {df['target'].nunique()} classes -> {path}")

save_sklearn(load_iris, "iris")
save_sklearn(load_wine, "wine")
save_sklearn(load_breast_cancer, "breast_cancer")

# Seeds: not in sklearn, pull from UCI
seeds_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
cols = ["area", "perimeter", "compactness", "kernel_length", "kernel_width",
        "asymmetry", "kernel_groove_length", "target"]
seeds = pd.read_csv(seeds_url, sep=r"\s+", header=None, names=cols)
seeds["target"] = seeds["target"] - 1  # UCI uses 1/2/3 -> make 0/1/2 for consistency
seeds.to_csv(f"{OUT}/seeds.csv", index=False)
print(f"seeds: {seeds.shape[0]} rows, {seeds.shape[1]-1} features, {seeds['target'].nunique()} classes -> {OUT}/seeds.csv")
