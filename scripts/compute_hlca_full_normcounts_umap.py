#!/usr/bin/env python3

import gc
import os
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import umap
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


H5AD = "/insomnia001/depts/morpheus/users/ob2391/Rstudio/batchelor/data/hlca_full_uncompressed.h5ad"
DATA_DIR = "/insomnia001/depts/morpheus/users/ob2391/Rstudio/batchelor/data"

COORD_OUTPUT = os.path.join(DATA_DIR, "hlca_full_normcounts_umap_coordinates.csv.gz")
HVG_OUTPUT = os.path.join(DATA_DIR, "hlca_full_normcounts_hvgs.csv")
SVD_VARIANCE_OUTPUT = os.path.join(DATA_DIR, "hlca_full_normcounts_svd_variance.csv")

RANDOM_STATE = 17
N_TOP_GENES = 3000
N_COMPONENTS = 50
N_NEIGHBORS = 30
MIN_DIST = 0.3
CHUNK_ROWS = 10000


def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def decode(values):
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return np.asarray(out, dtype=object)


def read_dataset_labels(h5):
    categories = decode(h5["obs/dataset/categories"][:])
    codes = h5["obs/dataset/codes"][:]
    labels = np.empty(codes.shape[0], dtype=object)
    labels[:] = None
    valid = codes >= 0
    labels[valid] = categories[codes[valid]]
    return labels


def choose_hvgs(h5):
    group = h5["X"]
    n_obs, n_vars = [int(x) for x in group.attrs["shape"]]
    data_ds = h5["X/data"]
    indices_ds = h5["X/indices"]
    indptr = h5["X/indptr"][:]

    col_sums = np.zeros(n_vars, dtype=np.float64)
    col_sqs = np.zeros(n_vars, dtype=np.float64)
    col_nnz = np.zeros(n_vars, dtype=np.int64)

    log(f"Streaming /X to estimate gene variance: {n_obs:,} cells x {n_vars:,} genes")
    for start in range(0, n_obs, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, n_obs)
        lo = int(indptr[start])
        hi = int(indptr[end])
        idx = indices_ds[lo:hi]
        dat = data_ds[lo:hi].astype(np.float64, copy=False)
        col_sums += np.bincount(idx, weights=dat, minlength=n_vars)
        col_sqs += np.bincount(idx, weights=dat * dat, minlength=n_vars)
        col_nnz += np.bincount(idx, minlength=n_vars)
        if start == 0 or end == n_obs or (start // CHUNK_ROWS) % 25 == 0:
            log(f"  variance pass rows {start:,}-{end:,}; nnz read={hi:,}")

    mean = col_sums / n_obs
    variance = np.maximum(col_sqs / n_obs - mean * mean, 0)
    dispersion = variance / np.maximum(mean, 1e-8)
    dispersion[col_nnz < 10] = -np.inf

    hvg_idx = np.argsort(dispersion)[-N_TOP_GENES:]
    hvg_idx = np.sort(hvg_idx.astype(np.int64))

    try:
        gene_names = decode(h5["var/_index"][:]).astype(str)
    except KeyError:
        gene_names = np.asarray([f"gene_{idx}" for idx in range(n_vars)], dtype=object)

    hvg_df = pd.DataFrame(
        {
            "gene_index": hvg_idx,
            "gene": gene_names[hvg_idx],
            "mean": mean[hvg_idx],
            "variance": variance[hvg_idx],
            "dispersion": dispersion[hvg_idx],
            "nonzero_cells": col_nnz[hvg_idx],
        }
    ).sort_values("dispersion", ascending=False)
    hvg_df.to_csv(HVG_OUTPUT, index=False)
    log(f"Wrote HVG table: {HVG_OUTPUT}")
    return hvg_idx, indptr


def build_hvg_matrix(h5, hvg_idx, indptr):
    group = h5["X"]
    n_obs, n_vars = [int(x) for x in group.attrs["shape"]]
    data_ds = h5["X/data"]
    indices_ds = h5["X/indices"]

    mapper = np.full(n_vars, -1, dtype=np.int32)
    mapper[hvg_idx] = np.arange(hvg_idx.size, dtype=np.int32)

    out_indptr = np.zeros(n_obs + 1, dtype=np.int64)
    data_chunks = []
    index_chunks = []
    cursor = 0

    log(f"Building sparse matrix restricted to {hvg_idx.size:,} HVGs")
    for start in range(0, n_obs, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, n_obs)
        ptr = indptr[start : end + 1]
        lo = int(ptr[0])
        hi = int(ptr[-1])
        idx = indices_ds[lo:hi]
        dat = data_ds[lo:hi]

        local = mapper[idx]
        keep = local >= 0

        row_counts = np.diff(ptr).astype(np.int64, copy=False)
        selected_counts = np.zeros(end - start, dtype=np.int64)
        nonzero_rows = row_counts > 0
        if hi > lo and np.any(nonzero_rows):
            starts = np.empty(end - start, dtype=np.int64)
            starts[0] = 0
            if starts.size > 1:
                starts[1:] = np.cumsum(row_counts[:-1])
            selected_counts[nonzero_rows] = np.add.reduceat(
                keep.astype(np.int32, copy=False),
                starts[nonzero_rows],
            )

        cumulative = np.cumsum(selected_counts)
        out_indptr[start + 1 : end + 1] = cursor + cumulative
        cursor += int(cumulative[-1]) if cumulative.size else 0

        if np.any(keep):
            data_chunks.append(dat[keep].astype(np.float32, copy=False))
            index_chunks.append(local[keep].astype(np.int32, copy=False))

        if start == 0 or end == n_obs or (start // CHUNK_ROWS) % 25 == 0:
            log(f"  HVG matrix rows {start:,}-{end:,}; selected nnz={cursor:,}")

    log(f"Concatenating selected sparse arrays; selected nnz={cursor:,}")
    data = np.concatenate(data_chunks).astype(np.float32, copy=False)
    indices = np.concatenate(index_chunks).astype(np.int32, copy=False)
    mat = sparse.csr_matrix((data, indices, out_indptr), shape=(n_obs, hvg_idx.size))
    mat.sort_indices()
    return mat


def main():
    if os.path.exists(COORD_OUTPUT):
        log(f"Coordinates already exist, not recomputing: {COORD_OUTPUT}")
        return

    with h5py.File(H5AD, "r") as h5:
        labels = read_dataset_labels(h5)
        hvg_idx, indptr = choose_hvgs(h5)
        mat = build_hvg_matrix(h5, hvg_idx, indptr)

    gc.collect()
    log(f"Running TruncatedSVD on normalized-count HVG matrix: shape={mat.shape}, nnz={mat.nnz:,}")
    svd_model = TruncatedSVD(
        n_components=N_COMPONENTS,
        algorithm="randomized",
        n_iter=3,
        random_state=RANDOM_STATE,
    )
    svd = svd_model.fit_transform(mat).astype(np.float32, copy=False)
    pd.DataFrame(
        {
            "component": np.arange(1, N_COMPONENTS + 1),
            "explained_variance": svd_model.explained_variance_,
            "explained_variance_ratio": svd_model.explained_variance_ratio_,
        }
    ).to_csv(SVD_VARIANCE_OUTPUT, index=False)
    log(f"Wrote SVD variance table: {SVD_VARIANCE_OUTPUT}")
    del mat
    gc.collect()

    log(f"Running UMAP on normalized-count SVD space: n_neighbors={N_NEIGHBORS}, min_dist={MIN_DIST}")
    reducer = umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        n_components=2,
        metric="euclidean",
        min_dist=MIN_DIST,
        low_memory=True,
        n_jobs=8,
        random_state=None,
        verbose=True,
    )
    coords = reducer.fit_transform(svd).astype(np.float32, copy=False)

    out = pd.DataFrame(
        {
            "UMAP_1": coords[:, 0],
            "UMAP_2": coords[:, 1],
            "dataset": labels.astype(str),
        }
    )
    out.to_csv(COORD_OUTPUT, index=False)
    log(f"Wrote normalized-count UMAP coordinates: {COORD_OUTPUT}")


if __name__ == "__main__":
    main()
