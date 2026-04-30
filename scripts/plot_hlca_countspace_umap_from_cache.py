#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


FIGURE_DIR = "/insomnia001/depts/morpheus/users/ob2391/Rstudio/batchelor/figures"
DATA_DIR = "/insomnia001/depts/morpheus/users/ob2391/Rstudio/batchelor/data"

PDF_OUTPUT = os.path.join(FIGURE_DIR, "hlca_core_countspace_umap_by_study_batch.pdf")
PER_STUDY_INPUT = os.path.join(FIGURE_DIR, "hlca_core_countspace_batch_silhouette_by_study.csv")
OVERALL_INPUT = os.path.join(FIGURE_DIR, "hlca_core_countspace_batch_silhouette_overall.csv")
COORD_INPUT = os.path.join(DATA_DIR, "hlca_core_countspace_umap_coordinates.csv.gz")

RANDOM_STATE = 1
SILHOUETTE_PER_BATCH = 1000


def set_umap_axes(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def main() -> None:
    coords = pd.read_csv(COORD_INPUT)
    per_study = pd.read_csv(PER_STUDY_INPUT)
    overall = pd.read_csv(OVERALL_INPUT)

    umap = coords[["UMAP_1", "UMAP_2"]].to_numpy(dtype=np.float32)
    study_labels = coords["study"].astype(str).to_numpy()
    counts_by_study = coords["study"].value_counts()
    study_levels = counts_by_study.index.to_numpy()
    overall_umap = float(overall.loc[overall["metric"] == "silhouette_umap", "mean_silhouette"].iloc[0])
    overall_svd = float(overall.loc[overall["metric"] == "silhouette_count_svd", "mean_silhouette"].iloc[0])
    sampled_n = int(overall["cells_sampled"].iloc[0])

    palette_values = [
        "#1F77B4", "#D62728", "#2CA02C", "#9467BD", "#FF7F0E", "#17BECF",
        "#8C564B", "#E377C2", "#BCBD22", "#111111", "#7F7F7F",
    ]
    palette = {level: palette_values[idx % len(palette_values)] for idx, level in enumerate(study_levels)}
    color_array = np.asarray([palette[label] for label in study_labels], dtype=object)
    rng = np.random.default_rng(RANDOM_STATE)
    order = rng.permutation(coords.shape[0])

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
                f"{coords.shape[0]:,} cells; /raw/X counts -> log-normalized HVGs -> sparse SVD -> UMAP; "
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

        fig, axes = plt.subplots(3, 4, figsize=(11, 8.5))
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
        fig.suptitle(
            "Raw-count-derived UMAP: one study highlighted per panel",
            fontsize=13,
            fontweight="bold",
            x=0.02,
            ha="left",
        )
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
                f"Silhouette sample n={sampled_n:,} "
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

    print(f"Wrote {PDF_OUTPUT} ({os.path.getsize(PDF_OUTPUT):,} bytes)")


if __name__ == "__main__":
    main()
