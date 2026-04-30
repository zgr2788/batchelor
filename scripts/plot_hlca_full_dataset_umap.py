#!/usr/bin/env python3

import argparse
import os

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors


H5AD = "/insomnia001/depts/morpheus/users/ob2391/Rstudio/batchelor/data/hlca_full_uncompressed.h5ad"
FIGURE_DIR = "/insomnia001/depts/morpheus/users/ob2391/Rstudio/batchelor/figures"
DATA_DIR = "/insomnia001/depts/morpheus/users/ob2391/Rstudio/batchelor/data"

RANDOM_STATE = 11
SILHOUETTE_MAX_TOTAL = 12000
KNN_SAMPLE_N = 50000
KNN_K = 30
TOP_LEVELS_IN_LEGEND = 48


def decode(values):
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return np.asarray(out, dtype=object)


def read_dataset_labels(h5ad_path):
    with h5py.File(h5ad_path, "r") as h5:
        categories = decode(h5["obs/dataset/categories"][:])
        codes = h5["obs/dataset/codes"][:]
    labels = np.empty(codes.shape[0], dtype=object)
    labels[:] = None
    valid = codes >= 0
    labels[valid] = categories[codes[valid]]
    return labels


def read_attached_umap(h5ad_path):
    with h5py.File(h5ad_path, "r") as h5:
        umap = h5["obsm/X_umap"][:]
    if umap.shape[0] == 2 and umap.shape[1] != 2:
        umap = umap.T
    if umap.shape[1] != 2:
        raise ValueError(f"Expected a two-column UMAP matrix, got {umap.shape}")
    return np.asarray(umap, dtype=np.float32)


def balanced_sample_indices(labels, rng, max_total):
    valid = pd.Series(labels).notna().to_numpy()
    labels_valid = labels[valid]
    valid_indices = np.flatnonzero(valid)
    counts = pd.Series(labels_valid).value_counts()
    counts = counts[counts >= 2]
    per_level = max(2, max_total // counts.shape[0])
    sampled = []
    for level in counts.index:
        idx = valid_indices[labels_valid == level]
        sampled.append(rng.choice(idx, size=min(idx.size, per_level), replace=False))
    sampled = np.concatenate(sampled)
    if sampled.size > max_total:
        sampled = rng.choice(sampled, size=max_total, replace=False)
    return sampled


def dataset_palette(levels):
    colors = []
    for cmap_name in ["tab20", "tab20b", "tab20c", "Set3", "Dark2", "Accent"]:
        cmap = plt.get_cmap(cmap_name)
        colors.extend([cmap(i) for i in range(getattr(cmap, "N", 20))])
    return {level: colors[i % len(colors)] for i, level in enumerate(levels)}


def set_umap_axes(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def compute_metrics(umap, labels, prefix):
    rng = np.random.default_rng(RANDOM_STATE)
    labels = labels.astype(object)
    counts = pd.Series(labels[pd.Series(labels).notna().to_numpy()]).value_counts()

    sample_idx = balanced_sample_indices(labels, rng, SILHOUETTE_MAX_TOTAL)
    sample_labels = labels[sample_idx].astype(str)
    sil = silhouette_samples(umap[sample_idx], sample_labels, metric="euclidean")

    sil_df = pd.DataFrame({"dataset": sample_labels, "silhouette": sil})
    per_level = (
        sil_df.groupby("dataset")
        .agg(
            silhouette_mean=("silhouette", "mean"),
            silhouette_median=("silhouette", "median"),
            cells_sampled=("dataset", "size"),
        )
        .reset_index()
    )
    per_level["cells_in_dataset"] = per_level["dataset"].map(counts).astype(int)
    per_level = per_level.sort_values("silhouette_mean", ascending=False)

    knn_sample = rng.choice(umap.shape[0], size=min(KNN_SAMPLE_N, umap.shape[0]), replace=False)
    nn = NearestNeighbors(n_neighbors=KNN_K + 1, metric="euclidean")
    nn.fit(umap)
    neighbor_idx = nn.kneighbors(umap[knn_sample], return_distance=False)[:, 1:]

    sample_label = labels[knn_sample]
    neighbor_labels = labels[neighbor_idx]
    sample_valid = pd.notna(sample_label)
    neighbor_valid = pd.notna(neighbor_labels)
    same = (neighbor_labels == sample_label[:, None]) & sample_valid[:, None] & neighbor_valid
    comparable = sample_valid[:, None] & neighbor_valid
    knn_same = float(same.sum() / comparable.sum())
    proportions = counts / counts.sum()
    expected_same = float(np.sum(np.square(proportions.to_numpy())))
    enrichment = float(knn_same / expected_same)

    overall = pd.DataFrame(
        [
            {
                "label": "dataset",
                "n_categories": int(counts.shape[0]),
                "n_cells": int(umap.shape[0]),
                "silhouette_mean": float(np.mean(sil)),
                "silhouette_median": float(np.median(sil)),
                "silhouette_sample_n": int(sample_idx.size),
                "knn_k": KNN_K,
                "knn_sample_n": int(knn_sample.size),
                "knn_same_label_fraction": knn_same,
                "knn_expected_same_label_fraction": expected_same,
                "knn_same_label_enrichment": enrichment,
            }
        ]
    )

    overall_path = os.path.join(FIGURE_DIR, f"{prefix}_dataset_metrics.csv")
    per_level_path = os.path.join(FIGURE_DIR, f"{prefix}_dataset_level_silhouettes.csv")
    overall.to_csv(overall_path, index=False)
    per_level.to_csv(per_level_path, index=False)
    return overall.iloc[0].to_dict(), per_level, counts, overall_path, per_level_path


def plot_umap(umap, labels, prefix, title, subtitle):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)
    overall, per_level, counts, overall_path, per_level_path = compute_metrics(umap, labels, prefix)

    levels = counts.index.to_list()
    palette = dataset_palette(levels)
    colors = np.asarray([palette.get(label, "#BDBDBD") for label in labels], dtype=object)
    order = rng.permutation(umap.shape[0])

    pdf_path = os.path.join(FIGURE_DIR, f"{prefix}_by_dataset.pdf")
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(12.5, 8.5))
        ax.scatter(
            umap[order, 0],
            umap[order, 1],
            c=colors[order],
            s=1.0,
            alpha=0.82,
            linewidths=0,
            rasterized=True,
        )
        set_umap_axes(ax)
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
        ax.text(
            0,
            1.01,
            (
                f"{subtitle}; dataset silhouette={overall['silhouette_mean']:.3f}; "
                f"kNN enrichment={overall['knn_same_label_enrichment']:.2f}x"
            ),
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
        )
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=palette[level],
                markersize=4.5,
                label=f"{level} (n={counts[level]:,})",
            )
            for level in levels[:TOP_LEVELS_IN_LEGEND]
        ]
        ax.legend(
            handles=handles,
            title="dataset",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=5.4,
            title_fontsize=8,
            frameon=False,
            ncol=1,
            handletextpad=0.3,
            borderaxespad=0,
        )
        fig.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        y = np.arange(per_level.shape[0])
        ax.axvline(0, color="#666666", linewidth=0.8)
        ax.barh(
            y,
            per_level["silhouette_mean"],
            color=[palette[level] for level in per_level["dataset"]],
        )
        ax.set_yticks(y)
        ax.set_yticklabels(per_level["dataset"], fontsize=5.5)
        ax.invert_yaxis()
        ax.set_xlabel("Mean silhouette width")
        ax.set_title("Per-dataset separability", fontsize=13, fontweight="bold", loc="left")
        ax.text(
            0,
            1.01,
            f"Balanced silhouette sample n={overall['silhouette_sample_n']:,}; kNN sample n={overall['knn_sample_n']:,}",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
        )
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Wrote {pdf_path}")
    print(f"Wrote {overall_path}")
    print(f"Wrote {per_level_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "attached",
            "normcounts",
            "corrected",
            "corrected_flann_kdtree",
            "corrected_flann_kmeans",
            "corrected_annoy",
            "corrected_nndescent",
            "corrected_rpforest",
        ],
        required=True,
        help=(
            "attached uses obsm/X_umap; normcounts uses cached normalized-count UMAP "
            "coordinates; corrected uses fastMNN+MlpackLshParam corrected UMAP coordinates; "
            "corrected_flann_kdtree uses fastMNN+FlannKdtreeParam corrected UMAP coordinates; "
            "corrected_flann_kmeans uses fastMNN+FlannKmeansParam corrected UMAP coordinates; "
            "corrected_annoy uses fastMNN+AnnoyParam corrected UMAP coordinates; "
            "corrected_nndescent uses fastMNN+NndescentParam corrected UMAP coordinates; "
            "corrected_rpforest uses fastMNN+RpforestParam corrected UMAP coordinates."
        ),
    )
    args = parser.parse_args()

    labels = read_dataset_labels(H5AD)
    if args.mode == "attached":
        umap = read_attached_umap(H5AD)
        plot_umap(
            umap,
            labels,
            "hlca_full_attached_umap",
            "HLCA full attached UMAP colored by dataset",
            "obsm/X_umap from HLCA full",
        )
    elif args.mode == "normcounts":
        coords_path = os.path.join(DATA_DIR, "hlca_full_normcounts_umap_coordinates.csv.gz")
        coords = pd.read_csv(coords_path)
        umap = coords[["UMAP_1", "UMAP_2"]].to_numpy(dtype=np.float32)
        plot_umap(
            umap,
            labels,
            "hlca_full_normcounts_umap",
            "HLCA full UMAP recalculated from normalized counts, colored by dataset",
            "/X normalized counts -> HVGs -> sparse SVD -> UMAP",
        )
    elif args.mode == "corrected":
        coords_path = os.path.join(
            DATA_DIR,
            "hlca_full_fastmnn_mlpack_lsh_corrected_umap_coordinates.csv.gz",
        )
        coords = pd.read_csv(coords_path)
        umap = coords[["UMAP_1", "UMAP_2"]].to_numpy(dtype=np.float32)
        plot_umap(
            umap,
            labels,
            "hlca_full_fastmnn_mlpack_lsh_corrected_umap",
            "HLCA full fastMNN corrected UMAP colored by dataset",
            "/X normalized counts HVGs -> fastMNN with MlpackLshParam -> UMAP",
        )
    elif args.mode == "corrected_flann_kdtree":
        coords_path = os.path.join(
            DATA_DIR,
            "hlca_full_fastmnn_flann_kdtree_corrected_umap_coordinates.csv.gz",
        )
        coords = pd.read_csv(coords_path)
        umap = coords[["UMAP_1", "UMAP_2"]].to_numpy(dtype=np.float32)
        plot_umap(
            umap,
            labels,
            "hlca_full_fastmnn_flann_kdtree_corrected_umap",
            "HLCA full fastMNN FLANN kd-tree corrected UMAP colored by dataset",
            "/X normalized counts HVGs -> fastMNN with FlannKdtreeParam -> UMAP",
        )
    elif args.mode == "corrected_flann_kmeans":
        coords_path = os.path.join(
            DATA_DIR,
            "hlca_full_fastmnn_flann_kmeans_corrected_umap_coordinates.csv.gz",
        )
        coords = pd.read_csv(coords_path)
        umap = coords[["UMAP_1", "UMAP_2"]].to_numpy(dtype=np.float32)
        plot_umap(
            umap,
            labels,
            "hlca_full_fastmnn_flann_kmeans_corrected_umap",
            "HLCA full fastMNN FLANN k-means corrected UMAP colored by dataset",
            "/X normalized counts HVGs -> fastMNN with FlannKmeansParam -> UMAP",
        )
    elif args.mode == "corrected_annoy":
        coords_path = os.path.join(
            DATA_DIR,
            "hlca_full_fastmnn_annoy_corrected_umap_coordinates.csv.gz",
        )
        coords = pd.read_csv(coords_path)
        umap = coords[["UMAP_1", "UMAP_2"]].to_numpy(dtype=np.float32)
        plot_umap(
            umap,
            labels,
            "hlca_full_fastmnn_annoy_corrected_umap",
            "HLCA full fastMNN Annoy corrected UMAP colored by dataset",
            "/X normalized counts HVGs -> fastMNN with AnnoyParam -> UMAP",
        )
    elif args.mode == "corrected_nndescent":
        coords_path = os.path.join(
            DATA_DIR,
            "hlca_full_fastmnn_nndescent_corrected_umap_coordinates.csv.gz",
        )
        coords = pd.read_csv(coords_path)
        umap = coords[["UMAP_1", "UMAP_2"]].to_numpy(dtype=np.float32)
        plot_umap(
            umap,
            labels,
            "hlca_full_fastmnn_nndescent_corrected_umap",
            "HLCA full fastMNN NN-descent corrected UMAP colored by dataset",
            "/X normalized counts HVGs -> fastMNN with NndescentParam -> UMAP",
        )
    else:
        coords_path = os.path.join(
            DATA_DIR,
            "hlca_full_fastmnn_rpforest_corrected_umap_coordinates.csv.gz",
        )
        coords = pd.read_csv(coords_path)
        umap = coords[["UMAP_1", "UMAP_2"]].to_numpy(dtype=np.float32)
        plot_umap(
            umap,
            labels,
            "hlca_full_fastmnn_rpforest_corrected_umap",
            "HLCA full fastMNN RP forest corrected UMAP colored by dataset",
            "/X normalized counts HVGs -> fastMNN with RpforestParam -> UMAP",
        )


if __name__ == "__main__":
    main()
