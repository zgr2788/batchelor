#!/usr/bin/env python3

import gc
import os
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy import sparse
from sklearn.metrics import silhouette_samples


INPUT = "/insomnia001/depts/morpheus/users/ob2391/Rstudio/batchelor/data/hlca_core_uncompressed.h5ad"
FIGURE_DIR = "/insomnia001/depts/morpheus/users/ob2391/Rstudio/batchelor/figures"
DATA_DIR = "/insomnia001/depts/morpheus/users/ob2391/Rstudio/batchelor/data"

PDF_OUTPUT = os.path.join(FIGURE_DIR, "hlca_core_countspace_umap_by_study_batch.pdf")
PER_STUDY_OUTPUT = os.path.join(FIGURE_DIR, "hlca_core_countspace_batch_silhouette_by_study.csv")
OVERALL_OUTPUT = os.path.join(FIGURE_DIR, "hlca_core_countspace_batch_silhouette_overall.csv")
COORD_OUTPUT = os.path.join(DATA_DIR, "hlca_core_countspace_umap_coordinates.csv.gz")

RANDOM_STATE = 1
N_TOP_GENES = 3000
N_COMPONENTS = 50
N_NEIGHBORS = 30
SILHOUETTE_PER_BATCH = 1000


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def decode_array(values):
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return np.asarray(out, dtype=object)


def read_categorical(h5, path):
    categories = decode_array(h5[f"{path}/categories"][:])
    codes = h5[f"{path}/codes"][:]
    labels = np.empty(codes.shape[0], dtype=object)
    labels[:] = None
    valid = codes >= 0
    labels[valid] = categories[codes[valid]]
    return pd.Categorical(labels, categories=categories)


def set_umap_axes(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def main() -> None:
    os.makedirs(FIGURE_DIR, exist_ok=True)

    log("Reading raw count CSR matrix from /raw/X")
    with h5py.File(INPUT, "r") as h5:
        shape = tuple(int(x) for x in h5["raw/X"].attrs["shape"])
        data = h5["raw/X/data"][:].astype(np.float32, copy=False)
        indices = h5["raw/X/indices"].astype("int32")[:]
        indptr = h5["raw/X/indptr"].astype("int32")[:]
        study = read_categorical(h5, "obs/study")
        try:
            var_names = decode_array(h5["raw/var/_index"][:])
        except KeyError:
            var_names = np.asarray([f"gene_{idx}" for idx in range(shape[1])], dtype=object)

    log(f"Constructing scipy CSR matrix: {shape[0]} cells x {shape[1]} genes; nnz={data.size}")
    counts = sparse.csr_matrix((data, indices, indptr), shape=shape)
    counts.sort_indices()
    del data, indices, indptr
    gc.collect()

    obs = pd.DataFrame({"study": study})
    obs.index = pd.Index([f"cell_{idx}" for idx in range(shape[0])], name="cell")
    var = pd.DataFrame(index=pd.Index(var_names.astype(str), name="gene"))
    adata = AnnData(X=counts, obs=obs, var=var)
    del counts
    gc.collect()

    log("Normalizing raw counts to 10,000 per cell and log1p-transforming")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    log(f"Selecting {N_TOP_GENES} highly variable genes from log-normalized counts")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=N_TOP_GENES,
        flavor="seurat",
        subset=True,
        check_values=False,
    )
    gc.collect()

    log(f"Running sparse count-space PCA/SVD with {N_COMPONENTS} components")
    sc.pp.pca(
        adata,
        n_comps=N_COMPONENTS,
        zero_center=False,
        svd_solver="randomized",
        random_state=RANDOM_STATE,
        dtype="float32",
        key_added="X_count_svd",
    )

    log(f"Building neighbors on count-space components; n_neighbors={N_NEIGHBORS}")
    sc.pp.neighbors(
        adata,
        n_neighbors=N_NEIGHBORS,
        use_rep="X_count_svd",
        metric="euclidean",
        random_state=RANDOM_STATE,
    )

    log("Computing UMAP from raw-count-derived neighbor graph")
    sc.tl.umap(adata, min_dist=0.3, spread=1.0, random_state=RANDOM_STATE)

    umap = np.asarray(adata.obsm["X_umap"], dtype=np.float32)
    svd = np.asarray(adata.obsm["X_count_svd"], dtype=np.float32)
    study_labels = adata.obs["study"].astype(str).to_numpy()
    counts_by_study = pd.Series(study_labels).value_counts()
    study_levels = counts_by_study.index.to_numpy()

    log("Calculating silhouette by study batch on count-space UMAP and count SVD")
    rng = np.random.default_rng(RANDOM_STATE)
    sampled_indices = []
    for level in study_levels:
        level_indices = np.flatnonzero(study_labels == level)
        n_take = min(level_indices.size, SILHOUETTE_PER_BATCH)
        sampled_indices.append(rng.choice(level_indices, size=n_take, replace=False))
    sampled_indices = np.concatenate(sampled_indices)
    sampled_labels = study_labels[sampled_indices]

    sil_umap = silhouette_samples(umap[sampled_indices], sampled_labels, metric="euclidean")
    sil_svd = silhouette_samples(svd[sampled_indices], sampled_labels, metric="euclidean")
    overall_umap = float(np.mean(sil_umap))
    overall_svd = float(np.mean(sil_svd))

    sil_df = pd.DataFrame(
        {
            "study": sampled_labels,
            "silhouette_umap": sil_umap,
            "silhouette_count_svd": sil_svd,
        }
    )
    per_study = (
        sil_df.groupby("study", observed=True)
        .agg(
            silhouette_umap=("silhouette_umap", "mean"),
            silhouette_count_svd=("silhouette_count_svd", "mean"),
            cells_sampled=("study", "size"),
        )
        .reset_index()
    )
    per_study["cells_in_batch"] = per_study["study"].map(counts_by_study).astype(int)
    per_study = per_study.sort_values("silhouette_umap", ascending=False)
    per_study.to_csv(PER_STUDY_OUTPUT, index=False)

    overall = pd.DataFrame(
        [
            {
                "metric": "silhouette_umap",
                "mean_silhouette": overall_umap,
                "cells_sampled": sampled_indices.size,
                "cells_total": adata.n_obs,
                "input": "/raw/X integer counts -> normalize_total -> log1p -> HVG -> sparse SVD -> UMAP",
            },
            {
                "metric": "silhouette_count_svd",
                "mean_silhouette": overall_svd,
                "cells_sampled": sampled_indices.size,
                "cells_total": adata.n_obs,
                "input": "/raw/X integer counts -> normalize_total -> log1p -> HVG -> sparse SVD",
            },
        ]
    )
    overall.to_csv(OVERALL_OUTPUT, index=False)

    coord_df = pd.DataFrame(
        {
            "UMAP_1": umap[:, 0],
            "UMAP_2": umap[:, 1],
            "study": study_labels,
        }
    )
    coord_df.to_csv(COORD_OUTPUT, index=False)

    log(f"Mean UMAP silhouette by study: {overall_umap:.4f}")
    log(f"Mean count-SVD silhouette by study: {overall_svd:.4f}")
    log(f"Wrote metrics: {PER_STUDY_OUTPUT}")
    log(f"Wrote coordinates: {COORD_OUTPUT}")

    palette_values = [
        "#1F77B4", "#D62728", "#2CA02C", "#9467BD", "#FF7F0E", "#17BECF",
        "#8C564B", "#E377C2", "#BCBD22", "#111111", "#7F7F7F",
    ]
    palette = {level: palette_values[idx % len(palette_values)] for idx, level in enumerate(study_levels)}
    color_array = np.asarray([palette[label] for label in study_labels], dtype=object)
    order = rng.permutation(adata.n_obs)

    log(f"Writing PDF: {PDF_OUTPUT}")
    with PdfPages(PDF_OUTPUT) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.scatter(
            umap[order, 0],
            umap[order, 1],
            c=color_array[order],
            s=1.8,
            alpha=0.88,
            linewidths=0,
            rasterized=True,
        )
        set_umap_axes(ax)
        ax.set_title(
            "HLCA core UMAP from raw counts, colored by study batch",
            fontsize=13,
            fontweight="bold",
            loc="left",
        )
        ax.text(
            0,
            1.01,
            (
                f"{adata.n_obs:,} cells; /raw/X counts -> log-normalized HVGs -> sparse SVD -> UMAP; "
                f"UMAP silhouette={overall_umap:.3f}; count-SVD silhouette={overall_svd:.3f}"
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
                markersize=6,
                label=f"{level} (n={counts_by_study[level]:,})",
            )
            for level in study_levels
        ]
        ax.legend(
            handles=handles,
            title="Study batch",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            fontsize=7,
            title_fontsize=9,
            frameon=False,
        )
        fig.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        fig, axes = plt.subplots(3, 4, figsize=(11, 8.5), sharex=True, sharey=True)
        axes = axes.ravel()
        for ax, level in zip(axes, study_levels):
            idx = study_labels == level
            ax.scatter(
                umap[order, 0],
                umap[order, 1],
                c="#D9D9D9",
                s=0.25,
                alpha=0.12,
                linewidths=0,
                rasterized=True,
            )
            level_order = rng.permutation(np.flatnonzero(idx))
            ax.scatter(
                umap[level_order, 0],
                umap[level_order, 1],
                c=palette[level],
                s=1.5,
                alpha=0.95,
                linewidths=0,
                rasterized=True,
            )
            set_umap_axes(ax)
            ax.set_title(f"{level}\nn={counts_by_study[level]:,}", fontsize=7)
        for ax in axes[len(study_levels):]:
            ax.axis("off")
        fig.suptitle("Raw-count-derived UMAP: one study highlighted per panel", fontsize=13, fontweight="bold", x=0.02, ha="left")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        count_plot = counts_by_study.sort_values(ascending=True)
        ax.barh(
            count_plot.index,
            count_plot.values,
            color=[palette[level] for level in count_plot.index],
        )
        ax.set_title("Cell counts by study batch", fontsize=13, fontweight="bold", loc="left")
        ax.set_xlabel("Cells")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(lambda x, pos: f"{int(x):,}")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        sil_plot = per_study.sort_values("silhouette_umap")
        y = np.arange(sil_plot.shape[0])
        ax.axvline(0, color="#666666", linewidth=0.8)
        ax.barh(
            y - 0.18,
            sil_plot["silhouette_umap"],
            height=0.34,
            color=[palette[level] for level in sil_plot["study"]],
            label="UMAP",
        )
        ax.barh(
            y + 0.18,
            sil_plot["silhouette_count_svd"],
            height=0.34,
            color="#BDBDBD",
            label="Count SVD",
        )
        ax.set_yticks(y)
        ax.set_yticklabels(sil_plot["study"])
        ax.set_xlabel("Mean silhouette width")
        ax.set_ylabel("")
        ax.set_title("Study-batch separability from raw counts", fontsize=13, fontweight="bold", loc="left")
        ax.text(
            0,
            1.01,
            (
                f"Silhouette sample n={sampled_indices.size:,} "
                f"({SILHOUETTE_PER_BATCH:,}/study); UMAP mean={overall_umap:.3f}; "
                f"count-SVD mean={overall_svd:.3f}"
            ),
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
        )
        ax.legend(frameon=False)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    log(f"Wrote PDF: {PDF_OUTPUT} ({os.path.getsize(PDF_OUTPUT):,} bytes)")


if __name__ == "__main__":
    main()
