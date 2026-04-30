#!/usr/bin/env python3

import argparse
import colorsys
import os

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import silhouette_samples


REPO_ROOT = "/insomnia001/depts/morpheus/users/ob2391/Rstudio/batchelor"
DATA_DIR = os.path.join(REPO_ROOT, "data")
FIGURE_DIR = os.path.join(REPO_ROOT, "figures")
H5AD = os.path.join(DATA_DIR, "hlca_full_uncompressed.h5ad")
RANDOM_STATE = 11
SILHOUETTE_MAX_TOTAL = 12000


BASE_PANELS = [
    {
        "id": "normcounts_umap",
        "title": "Normalized counts",
        "coord_path": os.path.join(DATA_DIR, "hlca_full_normcounts_umap_coordinates.csv.gz"),
        "coord_kind": "csv",
        "metric_path": os.path.join(FIGURE_DIR, "hlca_full_normcounts_umap_dataset_metrics.csv"),
    },
    {
        "id": "attached_umap",
        "title": "scANVI latent",
        "coord_path": "obsm/X_umap",
        "coord_kind": "h5ad_obsm",
        "metric_path": os.path.join(FIGURE_DIR, "hlca_full_attached_umap_dataset_metrics.csv"),
    },
]

BACKENDS = [
    {
        "id": "flann_kdtree",
        "metric_prefix": "fastmnn_flann_kdtree_corrected_umap",
        "title": "fastMNN FLANN kd-tree",
        "coord_path": os.path.join(DATA_DIR, "hlca_full_fastmnn_flann_kdtree_corrected_umap_coordinates.csv.gz"),
        "metric_path": os.path.join(
            FIGURE_DIR,
            "hlca_full_fastmnn_flann_kdtree_corrected_umap_dataset_metrics.csv",
        ),
    },
    {
        "id": "mlpack_lsh",
        "metric_prefix": "fastmnn_mlpack_lsh_corrected_umap",
        "title": "fastMNN mlpack LSH",
        "coord_path": os.path.join(DATA_DIR, "hlca_full_fastmnn_mlpack_lsh_corrected_umap_coordinates.csv.gz"),
        "metric_path": os.path.join(
            FIGURE_DIR,
            "hlca_full_fastmnn_mlpack_lsh_corrected_umap_dataset_metrics.csv",
        ),
    },
    {
        "id": "annoy",
        "metric_prefix": "fastmnn_annoy_corrected_umap",
        "title": "fastMNN Annoy",
        "coord_path": os.path.join(DATA_DIR, "hlca_full_fastmnn_annoy_corrected_umap_coordinates.csv.gz"),
        "metric_path": os.path.join(
            FIGURE_DIR,
            "hlca_full_fastmnn_annoy_corrected_umap_dataset_metrics.csv",
        ),
    },
]


COLOR_MODES = {
    "dataset": {
        "obs_key": "dataset",
        "legend_title": "Dataset",
        "metric_label": "dataset",
        "metric_title": "Dataset ASW",
        "output_token": "dataset",
    },
    "ann_level_1": {
        "obs_key": "ann_level_1",
        "legend_title": "Cell type",
        "metric_label": "ann_level_1",
        "metric_title": "Cell type ASW",
        "output_token": "ann_level_1",
    },
}


def decode(values):
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return np.asarray(out, dtype=object)


def read_categorical_labels(obs_key):
    with h5py.File(H5AD, "r") as h5:
        categories = decode(h5[f"obs/{obs_key}/categories"][:])
        codes = h5[f"obs/{obs_key}/codes"][:]
    labels = np.empty(codes.shape[0], dtype=object)
    labels[:] = None
    valid = codes >= 0
    labels[valid] = categories[codes[valid]]
    levels = [level for level in categories.tolist() if np.any(labels == level)]
    return labels, levels


def read_coordinates(panel):
    if panel["coord_kind"] == "csv":
        coords = pd.read_csv(panel["coord_path"], usecols=["UMAP_1", "UMAP_2"])
        return coords.to_numpy(dtype=np.float32)

    with h5py.File(H5AD, "r") as h5:
        coords = h5[panel["coord_path"]][:]
    if coords.shape[0] == 2 and coords.shape[1] != 2:
        coords = coords.T
    if coords.shape[1] != 2:
        raise ValueError(f"Expected two UMAP columns for {panel['coord_path']}, got {coords.shape}")
    return np.asarray(coords, dtype=np.float32)


def read_asw(panel):
    metrics = pd.read_csv(panel["metric_path"])
    return float(metrics.loc[0, "silhouette_mean"])


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


def computed_metric_path(panel, color_mode):
    return os.path.join(FIGURE_DIR, f"hlca_full_{panel['id']}_{color_mode['metric_label']}_metrics.csv")


def read_or_compute_asw(panel, coords, labels, color_mode):
    if color_mode["metric_label"] == "dataset":
        return read_asw(panel)

    metric_path = computed_metric_path(panel, color_mode)
    if os.path.exists(metric_path):
        metrics = pd.read_csv(metric_path)
        return float(metrics.loc[0, "silhouette_mean"])

    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = balanced_sample_indices(labels.astype(object), rng, SILHOUETTE_MAX_TOTAL)
    sample_labels = labels[sample_idx].astype(str)
    sil = silhouette_samples(coords[sample_idx], sample_labels, metric="euclidean")
    counts = pd.Series(labels[pd.Series(labels).notna().to_numpy()]).value_counts()
    metrics = pd.DataFrame(
        [
            {
                "label": color_mode["metric_label"],
                "n_categories": int(counts.shape[0]),
                "n_cells": int(coords.shape[0]),
                "silhouette_mean": float(np.mean(sil)),
                "silhouette_median": float(np.median(sil)),
                "silhouette_sample_n": int(sample_idx.size),
            }
        ]
    )
    metrics.to_csv(metric_path, index=False)
    print(f"Wrote {metric_path}")
    return float(metrics.loc[0, "silhouette_mean"])


def distinct_palette(levels):
    seed = [
        "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9",
        "#E69F00", "#000000", "#332288", "#88CCEE", "#44AA99", "#117733",
        "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499", "#6699CC",
        "#661100", "#669900", "#AA4466", "#4477AA", "#228833", "#CCBB44",
        "#EE6677", "#AA3377", "#BBBBBB", "#33BBEE", "#EE7733", "#EE3377",
        "#BBBB33", "#77AADD", "#99DDFF", "#44BB99", "#BBCC33", "#AAAA00",
        "#EEDD88", "#EE8866", "#FFAABB", "#DDDDDD", "#1B9E77", "#D95F02",
        "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#666666",
    ]
    if len(levels) <= len(seed):
        colors = seed[: len(levels)]
    else:
        colors = seed.copy()
        hue = 0.0
        while len(colors) < len(levels):
            hue = (hue + 0.618033988749895) % 1.0
            sat = 0.66 + 0.18 * ((len(colors) // 6) % 2)
            light = 0.42 + 0.11 * ((len(colors) // 3) % 3)
            rgb = colorsys.hls_to_rgb(hue, light, sat)
            colors.append("#%02x%02x%02x" % tuple(int(round(v * 255)) for v in rgb))
    if len(set(colors)) != len(colors):
        raise ValueError("Dataset palette contains duplicate colors")
    return dict(zip(levels, colors))


def normalize_for_panel(coords):
    out = coords.copy()
    out[:, 0] -= np.nanmedian(out[:, 0])
    out[:, 1] -= np.nanmedian(out[:, 1])
    span = max(np.nanmax(out[:, 0]) - np.nanmin(out[:, 0]), np.nanmax(out[:, 1]) - np.nanmin(out[:, 1]))
    if span > 0:
        out /= span
    return out


def style_axis(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1", labelpad=2)
    ax.set_ylabel("UMAP 2", labelpad=2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.45)


def draw_panel(ax, panel, letter, colors, order, labels, color_mode):
    coords = normalize_for_panel(read_coordinates(panel))
    asw = read_or_compute_asw(panel, coords, labels, color_mode)
    ax.scatter(
        coords[order, 0],
        coords[order, 1],
        c=colors[order],
        s=0.18,
        alpha=0.80,
        linewidths=0,
        rasterized=True,
    )
    style_axis(ax)
    ax.set_title(
        f"{panel['title']}\n{color_mode['metric_title']} = {asw:.3f}",
        loc="left",
        fontsize=9.2,
        fontweight="bold",
        pad=4,
    )
    ax.text(
        -0.075,
        1.07,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def render_backend_figure(backend, colors, order, palette, levels, labels, color_mode):
    panels = BASE_PANELS + [
        {
            "id": backend["metric_prefix"],
            "title": backend["title"],
            "coord_path": backend["coord_path"],
            "coord_kind": "csv",
            "metric_path": backend["metric_path"],
        }
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 6.0), constrained_layout=False)
    fig.subplots_adjust(left=0.032, right=0.995, top=0.84, bottom=0.31, wspace=0.055)

    for ax, panel, letter in zip(axes, panels, ["a", "b", "c"]):
        draw_panel(ax, panel, letter, colors, order, labels, color_mode)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=palette[level],
            markeredgewidth=0,
            markersize=3.2,
            alpha=0.95,
            label=level,
        )
        for level in levels
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=4,
        frameon=False,
        fontsize=5.0,
        handletextpad=0.35,
        columnspacing=1.0,
        borderaxespad=0,
        title=color_mode["legend_title"],
        title_fontsize=5.5,
    )

    output_pdf = os.path.join(
        FIGURE_DIR,
        (
            "hlca_full_umap_"
            f"{color_mode['output_token']}_comparison_normcounts_scanvi_fastmnn_{backend['id']}.pdf"
        ),
    )
    output_png = output_pdf.replace(".pdf", ".png")
    fig.savefig(output_pdf, dpi=500, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output_pdf}")
    print(f"Wrote {output_png}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--color-modes",
        nargs="*",
        choices=list(COLOR_MODES),
        default=list(COLOR_MODES),
        help="Label sets to use for point colors. Defaults to dataset and ann_level_1.",
    )
    parser.add_argument(
        "--backends",
        nargs="*",
        choices=[backend["id"] for backend in BACKENDS],
        default=[backend["id"] for backend in BACKENDS],
        help="Backend figures to render. Defaults to all saved backend UMAPs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(FIGURE_DIR, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    requested = set(args.backends)
    for color_mode_id in args.color_modes:
        color_mode = COLOR_MODES[color_mode_id]
        labels, levels = read_categorical_labels(color_mode["obs_key"])
        palette = distinct_palette(levels)
        if len(set(palette.values())) != len(levels):
            raise ValueError(f"Duplicate colors detected for {color_mode_id}")

        palette_path = os.path.join(FIGURE_DIR, f"hlca_full_{color_mode['output_token']}_palette.csv")
        pd.DataFrame({"label": levels, "color": [palette[level] for level in levels]}).to_csv(
            palette_path,
            index=False,
        )
        print(f"Wrote {palette_path}")

        colors = np.asarray([palette.get(label, "#BDBDBD") for label in labels], dtype=object)
        rng = np.random.default_rng(RANDOM_STATE)
        order = rng.permutation(labels.shape[0])

        for backend in BACKENDS:
            if backend["id"] in requested:
                render_backend_figure(backend, colors, order, palette, levels, labels, color_mode)


if __name__ == "__main__":
    main()
