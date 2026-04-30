#!/usr/bin/env python3

import math
import os
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors


H5AD_INPUT = "/insomnia001/depts/morpheus/users/ob2391/Rstudio/batchelor/data/hlca_core_uncompressed.h5ad"
COORD_INPUT = "/insomnia001/depts/morpheus/users/ob2391/Rstudio/batchelor/data/hlca_core_countspace_umap_coordinates.csv.gz"
FIGURE_DIR = "/insomnia001/depts/morpheus/users/ob2391/Rstudio/batchelor/figures"

PDF_OUTPUT = os.path.join(FIGURE_DIR, "hlca_core_countspace_umap_batch_candidate_scan.pdf")
METRICS_OUTPUT = os.path.join(FIGURE_DIR, "hlca_core_countspace_batch_candidate_metrics.csv")
PER_LEVEL_OUTPUT = os.path.join(FIGURE_DIR, "hlca_core_countspace_batch_candidate_level_silhouettes.csv")

RANDOM_STATE = 7
SILHOUETTE_MAX_TOTAL = 12000
KNN_SAMPLE_N = 50000
KNN_K = 30
TOP_PLOT_CANDIDATES = 8
TOP_ENRICHMENT_PLOT_CANDIDATES = 6
TOP_LEVELS_IN_LEGEND = 20

# Excludes obvious biological annotations such as cell type, disease, tissue,
# sex, ancestry, age/development, smoking, and annotation/cluster labels.
CANDIDATE_COLUMNS = [
    "dataset",
    "study",
    "sample",
    "donor_id",
    "assay",
    "assay_ontology_term_id",
    "sequencing_platform",
    "tissue_dissociation_protocol",
    "tissue_sampling_method",
    "fresh_or_frozen",
    "suspension_type",
    "reference_genome",
    "subject_type",
    "reannotation_type",
]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def decode(values):
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return np.asarray(out, dtype=object)


def read_obs_column(h5, column: str) -> pd.Series:
    path = f"obs/{column}"
    node = h5[path]
    if isinstance(node, h5py.Group) and "categories" in node and "codes" in node:
        categories = decode(node["categories"][:])
        codes = node["codes"][:]
        labels = np.empty(codes.shape[0], dtype=object)
        labels[:] = None
        valid = codes >= 0
        labels[valid] = categories[codes[valid]]
        return pd.Series(pd.Categorical(labels, categories=categories), name=column)

    values = node[:]
    if values.dtype.kind in {"S", "O", "U"}:
        return pd.Series(decode(values), name=column).astype("category")
    return pd.Series(values, name=column)


def balanced_sample_indices(labels: np.ndarray, rng: np.random.Generator, max_total: int) -> np.ndarray:
    valid = pd.Series(labels).notna().to_numpy()
    labels_valid = labels[valid]
    valid_indices = np.flatnonzero(valid)
    counts = pd.Series(labels_valid).value_counts()
    counts = counts[counts >= 2]
    if counts.shape[0] < 2:
        return np.array([], dtype=np.int64)

    per_level = max(2, max_total // counts.shape[0])
    sampled = []
    for level in counts.index:
        idx = valid_indices[labels_valid == level]
        n_take = min(idx.size, per_level)
        sampled.append(rng.choice(idx, size=n_take, replace=False))
    sampled = np.concatenate(sampled)
    if sampled.size > max_total:
        sampled = rng.choice(sampled, size=max_total, replace=False)
    return sampled


def candidate_palette(levels: list[str]) -> dict[str, str]:
    cmap_names = ["tab20", "tab20b", "tab20c", "Set3", "Dark2", "Accent"]
    colors = []
    for name in cmap_names:
        cmap = plt.get_cmap(name)
        n = getattr(cmap, "N", 20)
        colors.extend([cmap(i) for i in range(n)])
    return {level: colors[i % len(colors)] for i, level in enumerate(levels)}


def set_umap_axes(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_candidate(ax, umap, labels, column, metric_row, rng, base_order):
    series = pd.Series(labels).fillna("NA").astype(str)
    counts = series.value_counts()
    if counts.shape[0] > TOP_LEVELS_IN_LEGEND:
        visible_levels = counts.head(TOP_LEVELS_IN_LEGEND).index.tolist()
        plot_labels = series.where(series.isin(visible_levels), other="Other").to_numpy()
        legend_levels = visible_levels + ["Other"]
    else:
        legend_levels = counts.index.tolist()
        plot_labels = series.to_numpy()

    palette = candidate_palette(legend_levels)
    if "Other" in palette:
        palette["Other"] = "#CFCFCF"
    colors = np.asarray([palette[label] for label in plot_labels], dtype=object)

    ax.scatter(
        umap[base_order, 0],
        umap[base_order, 1],
        c=colors[base_order],
        s=1.9,
        alpha=0.88,
        linewidths=0,
        rasterized=True,
    )
    set_umap_axes(ax)
    ax.set_title(
        f"{column}: silhouette={metric_row['silhouette_mean']:.3f}, "
        f"kNN enrich={metric_row['knn_same_label_enrichment']:.2f}x",
        fontsize=10,
        fontweight="bold",
        loc="left",
    )
    subtitle = f"{metric_row['n_categories']} categories; n={int(metric_row['n_valid']):,}"
    if counts.shape[0] > TOP_LEVELS_IN_LEGEND:
        subtitle += f"; showing top {TOP_LEVELS_IN_LEGEND}, rest grey"
    ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=8, va="bottom")

    handles = []
    for level in legend_levels[:TOP_LEVELS_IN_LEGEND + 1]:
        label_count = int((plot_labels == level).sum())
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=palette[level],
                markersize=5,
                label=f"{level} (n={label_count:,})",
            )
        )
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=5.5,
        frameon=False,
        handletextpad=0.3,
        borderaxespad=0,
    )


def main() -> None:
    os.makedirs(FIGURE_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    log("Reading count-space UMAP coordinates")
    coords = pd.read_csv(COORD_INPUT)
    umap = coords[["UMAP_1", "UMAP_2"]].to_numpy(dtype=np.float32)
    n_cells = umap.shape[0]
    base_order = rng.permutation(n_cells)

    log("Preparing fixed kNN sample on count-space UMAP")
    knn_sample = rng.choice(n_cells, size=min(KNN_SAMPLE_N, n_cells), replace=False)
    nn = NearestNeighbors(n_neighbors=KNN_K + 1, metric="euclidean", algorithm="auto")
    nn.fit(umap)
    neighbor_idx = nn.kneighbors(umap[knn_sample], return_distance=False)[:, 1:]

    all_labels = {}
    with h5py.File(H5AD_INPUT, "r") as h5:
        for column in CANDIDATE_COLUMNS:
            if f"obs/{column}" not in h5:
                log(f"Skipping missing column: {column}")
                continue
            values = read_obs_column(h5, column)
            if not isinstance(values.dtype, pd.CategoricalDtype) and values.nunique(dropna=True) > 200:
                log(f"Skipping non-categorical/high-cardinality column: {column}")
                continue
            all_labels[column] = values.astype(object).where(values.notna(), None).to_numpy()

    metrics = []
    per_level_rows = []
    for column, labels in all_labels.items():
        valid_mask = pd.Series(labels).notna().to_numpy()
        valid_labels = labels[valid_mask]
        counts = pd.Series(valid_labels).value_counts()
        counts = counts[counts >= 2]
        if counts.shape[0] < 2:
            log(f"Skipping {column}: fewer than two usable levels")
            continue

        sample_idx = balanced_sample_indices(labels, rng, SILHOUETTE_MAX_TOTAL)
        sample_labels = labels[sample_idx].astype(str)
        sil = silhouette_samples(umap[sample_idx], sample_labels, metric="euclidean")
        sil_mean = float(np.mean(sil))
        sil_median = float(np.median(sil))

        sample_df = pd.DataFrame({"level": sample_labels, "silhouette": sil})
        level_sil = (
            sample_df.groupby("level")
            .agg(silhouette_mean=("silhouette", "mean"), silhouette_median=("silhouette", "median"), cells_sampled=("level", "size"))
            .reset_index()
        )
        level_sil["column"] = column
        level_sil["cells_in_level"] = level_sil["level"].map(counts).astype(int)
        per_level_rows.append(level_sil)

        labels_full = labels.astype(object)
        sample_label = labels_full[knn_sample]
        neighbor_labels = labels_full[neighbor_idx]
        sample_valid = pd.notna(sample_label)
        neighbor_valid = pd.notna(neighbor_labels)
        same = neighbor_labels == sample_label[:, None]
        same = same & sample_valid[:, None] & neighbor_valid
        comparable = sample_valid[:, None] & neighbor_valid
        knn_same = float(same.sum() / comparable.sum())
        proportions = counts / counts.sum()
        expected_same = float(np.sum(np.square(proportions.to_numpy())))
        enrichment = float(knn_same / expected_same) if expected_same > 0 else math.nan

        metrics.append(
            {
                "column": column,
                "n_categories": int(counts.shape[0]),
                "n_valid": int(valid_mask.sum()),
                "n_missing": int((~valid_mask).sum()),
                "silhouette_mean": sil_mean,
                "silhouette_median": sil_median,
                "silhouette_sample_n": int(sample_idx.size),
                "knn_same_label_fraction": knn_same,
                "knn_expected_same_label_fraction": expected_same,
                "knn_same_label_enrichment": enrichment,
                "largest_level": str(counts.index[0]),
                "largest_level_cells": int(counts.iloc[0]),
            }
        )
        log(
            f"{column}: silhouette={sil_mean:.4f}; "
            f"kNN same={knn_same:.3f}; expected={expected_same:.3f}; enrich={enrichment:.2f}x"
        )

    metrics_df = pd.DataFrame(metrics).sort_values("silhouette_mean", ascending=False)
    metrics_df.to_csv(METRICS_OUTPUT, index=False)
    per_level_df = pd.concat(per_level_rows, ignore_index=True)
    per_level_df = per_level_df.sort_values(["column", "silhouette_mean"], ascending=[True, False])
    per_level_df.to_csv(PER_LEVEL_OUTPUT, index=False)
    log(f"Wrote metrics: {METRICS_OUTPUT}")
    log(f"Wrote per-level metrics: {PER_LEVEL_OUTPUT}")

    top_columns = metrics_df.head(TOP_PLOT_CANDIDATES)["column"].tolist()
    top_enrichment_columns = (
        metrics_df.sort_values("knn_same_label_enrichment", ascending=False)
        .head(TOP_ENRICHMENT_PLOT_CANDIDATES)["column"]
        .tolist()
    )
    top_columns = list(dict.fromkeys(top_columns + top_enrichment_columns))
    log(f"Plotting top candidates: {', '.join(top_columns)}")
    with PdfPages(PDF_OUTPUT) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        y = np.arange(metrics_df.shape[0])
        ax.axvline(0, color="#666666", linewidth=0.8)
        ax.barh(y, metrics_df["silhouette_mean"], color="#3B6FB6")
        ax.set_yticks(y)
        ax.set_yticklabels(metrics_df["column"])
        ax.invert_yaxis()
        ax.set_xlabel("Mean silhouette width on count-space UMAP")
        ax.set_title("Batch-candidate separability scan", fontsize=13, fontweight="bold", loc="left")
        ax.text(
            0,
            1.01,
            f"Silhouette sampled up to {SILHOUETTE_MAX_TOTAL:,} cells per column; biological columns omitted",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
        )
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        enrich_sorted = metrics_df.sort_values("knn_same_label_enrichment", ascending=False)
        y = np.arange(enrich_sorted.shape[0])
        ax.barh(y, enrich_sorted["knn_same_label_enrichment"], color="#C44E52")
        ax.set_yticks(y)
        ax.set_yticklabels(enrich_sorted["column"])
        ax.invert_yaxis()
        ax.set_xlabel("kNN same-label enrichment over expected")
        ax.set_title("Local same-label enrichment on count-space UMAP", fontsize=13, fontweight="bold", loc="left")
        ax.text(
            0,
            1.01,
            f"k={KNN_K}; query sample n={min(KNN_SAMPLE_N, n_cells):,}; higher means local neighborhoods share this label more than expected",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
        )
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        for column in top_columns:
            metric_row = metrics_df.loc[metrics_df["column"] == column].iloc[0]
            fig, ax = plt.subplots(figsize=(11, 8.5))
            plot_candidate(ax, umap, all_labels[column], column, metric_row, rng, base_order)
            fig.tight_layout()
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

    log(f"Wrote PDF: {PDF_OUTPUT} ({os.path.getsize(PDF_OUTPUT):,} bytes)")


if __name__ == "__main__":
    main()
