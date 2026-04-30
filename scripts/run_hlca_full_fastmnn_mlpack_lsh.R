#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(rhdf5)
    library(Matrix)
    library(batchelor)
    library(BiocNeighbors)
    library(BiocParallel)
    library(BiocSingular)
    library(SingleCellExperiment)
    library(uwot)
})

repo_root <- "/insomnia001/depts/morpheus/users/ob2391/Rstudio/batchelor"
h5ad_path <- file.path(repo_root, "data", "hlca_full_uncompressed.h5ad")
hvg_path <- file.path(repo_root, "data", "hlca_full_normcounts_hvgs.csv")
data_dir <- file.path(repo_root, "data")

chunk_rows <- 10000L
workers <- 128L
max_cells <- NA_integer_
method_id <- "rpforest"
force_preprocess <- FALSE

args <- commandArgs(trailingOnly=TRUE)
for (arg in args) {
    if (grepl("^--workers=", arg)) {
        workers <- as.integer(sub("^--workers=", "", arg))
    } else if (grepl("^--max-cells=", arg)) {
        max_cells <- as.integer(sub("^--max-cells=", "", arg))
    } else if (grepl("^--chunk-rows=", arg)) {
        chunk_rows <- as.integer(sub("^--chunk-rows=", "", arg))
    } else if (grepl("^--method=", arg)) {
        method_id <- sub("^--method=", "", arg)
    } else if (arg == "--force-preprocess") {
        force_preprocess <- TRUE
    } else {
        stop("Unknown argument: ", arg)
    }
}

method_label <- switch(
    method_id,
    mlpack_lsh="MlpackLshParam",
    rpforest="RpforestParam",
    flann_kdtree="FlannKdtreeParam",
    flann_kmeans="FlannKmeansParam",
    annoy="AnnoyParam",
    nndescent="NndescentParam",
    stop("Unsupported method: ", method_id)
)

make_bnparam <- function(method_id) {
    switch(
        method_id,
        mlpack_lsh=BiocNeighbors::MlpackLshParam(),
        rpforest=BiocNeighbors::RpforestParam(),
        flann_kdtree=BiocNeighbors::FlannKdtreeParam(),
        flann_kmeans=BiocNeighbors::FlannKmeansParam(),
        annoy=BiocNeighbors::AnnoyParam(),
        nndescent=BiocNeighbors::NndescentParam(),
        stop("Unsupported method: ", method_id)
    )
}

output_prefix <- paste0("hlca_full_fastmnn_", method_id)
if (!is.na(max_cells)) {
    output_prefix <- paste0(output_prefix, "_smoke_", max_cells)
}
corrected_rds <- file.path(data_dir, paste0(output_prefix, "_corrected_coords.rds"))
umap_csv <- file.path(data_dir, paste0(output_prefix, "_corrected_umap_coordinates.csv.gz"))
merge_info_rds <- file.path(data_dir, paste0(output_prefix, "_merge_info.rds"))
preprocess_suffix <- if (is.na(max_cells)) "" else paste0("_smoke_", max_cells)
preprocess_cache <- file.path(data_dir, paste0("hlca_full_normcounts_hvg_fastmnn_preprocessed", preprocess_suffix, ".rds"))

log_message <- function(...) {
    message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " ", paste0(..., collapse=""))
}

current_rss_gb <- function() {
    status <- tryCatch(readLines("/proc/self/status"), error=function(e) character())
    rss <- grep("^VmRSS:", status, value=TRUE)
    if (!length(rss)) {
        return(NA_real_)
    }
    as.numeric(gsub("[^0-9]", "", rss)) / 1024^2
}

read_h5_slice <- function(name, start, count) {
    h5read(
        h5ad_path,
        name,
        start=as.numeric(start),
        count=as.numeric(count),
        read.attributes=FALSE,
        bit64conversion="double"
    )
}

read_dataset_labels <- function(n_cells) {
    cats <- h5read(h5ad_path, "/obs/dataset/categories", read.attributes=FALSE)
    codes <- h5read(h5ad_path, "/obs/dataset/codes", read.attributes=FALSE)
    if (!is.na(n_cells)) {
        codes <- codes[seq_len(n_cells)]
    }
    labels <- rep(NA_character_, length(codes))
    valid <- codes >= 0L
    labels[valid] <- cats[codes[valid] + 1L]
    labels
}

count_hvg_entries <- function(indptr, hvg_indices0, n_vars, n_cells) {
    mapper <- rep.int(NA_integer_, n_vars)
    mapper[hvg_indices0 + 1L] <- seq_along(hvg_indices0) - 1L

    p <- integer(n_cells + 1L)
    cursor <- 0L

    for (start_cell in seq.int(1L, n_cells, by=chunk_rows)) {
        end_cell <- min(start_cell + chunk_rows - 1L, n_cells)
        ptr <- indptr[start_cell:(end_cell + 1L)]
        lo <- ptr[1L]
        hi <- ptr[length(ptr)]
        count <- hi - lo

        selected_counts <- integer(end_cell - start_cell + 1L)
        if (count > 0) {
            idx <- as.integer(read_h5_slice("/X/indices", lo + 1, count))
            keep <- !is.na(mapper[idx + 1L])
            row_counts <- as.integer(diff(ptr))
            nonzero <- row_counts > 0L
            if (any(nonzero)) {
                row_ids <- rep.int(seq_along(row_counts), row_counts)
                selected_counts <- tabulate(row_ids[keep], nbins=length(row_counts))
            }
        }

        cumulative <- cumsum(selected_counts)
        p[(start_cell + 1L):(end_cell + 1L)] <- cursor + cumulative
        cursor <- cursor + if (length(cumulative)) cumulative[length(cumulative)] else 0L

        if (start_cell == 1L || end_cell == n_cells || ((start_cell - 1L) / chunk_rows) %% 25L == 0L) {
            log_message("count pass cells ", format(start_cell, big.mark=","), "-",
                format(end_cell, big.mark=","), "; selected nnz=", format(cursor, big.mark=","),
                "; RSS=", sprintf("%.2f GiB", current_rss_gb()))
        }
    }

    list(p=p, mapper=mapper, nnz=cursor)
}

fill_hvg_matrix <- function(indptr, p, mapper, n_hvgs, n_cells) {
    nnz <- p[length(p)]
    log_message("allocating sparse slots for selected nnz=", format(nnz, big.mark=","))
    i <- integer(nnz)
    x <- numeric(nnz)

    cursor <- 1L
    for (start_cell in seq.int(1L, n_cells, by=chunk_rows)) {
        end_cell <- min(start_cell + chunk_rows - 1L, n_cells)
        ptr <- indptr[start_cell:(end_cell + 1L)]
        lo <- ptr[1L]
        hi <- ptr[length(ptr)]
        count <- hi - lo

        if (count > 0) {
            idx <- as.integer(read_h5_slice("/X/indices", lo + 1, count))
            dat <- read_h5_slice("/X/data", lo + 1, count)
            mapped <- mapper[idx + 1L]
            keep <- !is.na(mapped)
            n_keep <- sum(keep)
            if (n_keep > 0L) {
                target <- cursor:(cursor + n_keep - 1L)
                i[target] <- mapped[keep]
                x[target] <- dat[keep]
                cursor <- cursor + n_keep
            }
        }

        if (start_cell == 1L || end_cell == n_cells || ((start_cell - 1L) / chunk_rows) %% 25L == 0L) {
            log_message("fill pass cells ", format(start_cell, big.mark=","), "-",
                format(end_cell, big.mark=","), "; filled nnz=", format(cursor - 1L, big.mark=","),
                "; RSS=", sprintf("%.2f GiB", current_rss_gb()))
        }
    }

    mat <- new(
        "dgCMatrix",
        i=i,
        p=p,
        x=x,
        Dim=as.integer(c(n_hvgs, n_cells))
    )
    validObject(mat)
    mat
}

write_umap_csv <- function(coords, labels, path) {
    con <- gzfile(path, open="wt")
    on.exit(close(con))
    writeLines("UMAP_1,UMAP_2,dataset", con)
    chunk <- 100000L
    n <- nrow(coords)
    for (start in seq.int(1L, n, by=chunk)) {
        end <- min(start + chunk - 1L, n)
        out <- data.frame(
            UMAP_1=coords[start:end, 1],
            UMAP_2=coords[start:end, 2],
            dataset=labels[start:end],
            check.names=FALSE
        )
        utils::write.table(out, con, sep=",", row.names=FALSE, col.names=FALSE, quote=TRUE)
    }
}

if (file.exists(umap_csv)) {
    log_message("UMAP coordinate output already exists, not recomputing: ", umap_csv)
    quit(save="no", status=0L)
}

Sys.setenv(HDF5_USE_FILE_LOCKING="FALSE")
log_message("method=", method_label, "; workers=", workers, "; chunk_rows=", chunk_rows, "; max_cells=", ifelse(is.na(max_cells), "all", max_cells))
log_message("input=", h5ad_path)
log_message("preprocess cache=", preprocess_cache)

if (file.exists(preprocess_cache) && !force_preprocess) {
    log_message("loading cached non-backend preprocessing")
    cached <- readRDS(preprocess_cache)
    mat <- cached$mat
    labels <- cached$labels
    counts <- cached$counts
    batch <- factor(labels, levels=cached$batch_levels)
    n_cells <- cached$n_cells
    n_vars <- cached$n_vars
    n_hvgs <- cached$n_hvgs
    log_message("loaded cached sparse HVG matrix: ", paste(dim(mat), collapse=" x "),
        "; nnz=", format(length(mat@x), big.mark=","),
        "; RSS=", sprintf("%.2f GiB", current_rss_gb()))
} else {
    if (force_preprocess && file.exists(preprocess_cache)) {
        log_message("force-preprocess requested; rebuilding cache")
    } else {
        log_message("preprocess cache missing; building sparse HVG matrix from H5AD")
    }

    attrs <- h5readAttributes(h5ad_path, "/X")
    shape <- as.integer(attrs$shape)
    n_cells <- shape[1L]
    n_vars <- shape[2L]
    if (!is.na(max_cells)) {
        n_cells <- min(n_cells, max_cells)
    }

    hvgs <- utils::read.csv(hvg_path, stringsAsFactors=FALSE)
    hvgs <- hvgs[order(hvgs$gene_index), , drop=FALSE]
    hvg_indices0 <- as.integer(hvgs$gene_index)
    n_hvgs <- length(hvg_indices0)

    labels <- read_dataset_labels(n_cells)
    counts <- sort(table(labels), decreasing=TRUE)
    batch <- factor(labels, levels=names(counts))

    log_message("cells=", format(n_cells, big.mark=","), "; vars=", format(n_vars, big.mark=","),
        "; HVGs=", format(n_hvgs, big.mark=","), "; batches=", length(counts))
    log_message("largest batches: ", paste(head(paste0(names(counts), "=", as.integer(counts)), 8L), collapse=", "))

    indptr <- h5read(h5ad_path, "/X/indptr", read.attributes=FALSE, bit64conversion="double")
    if (!is.na(max_cells)) {
        indptr <- indptr[seq_len(n_cells + 1L)]
    }

    counted <- count_hvg_entries(indptr, hvg_indices0, n_vars, n_cells)
    mat <- fill_hvg_matrix(indptr, counted$p, counted$mapper, n_hvgs, n_cells)
    rownames(mat) <- hvgs$gene
    rm(counted, indptr)
    invisible(gc())
    log_message("sparse HVG matrix constructed: ", paste(dim(mat), collapse=" x "),
        "; nnz=", format(length(mat@x), big.mark=","), "; RSS=", sprintf("%.2f GiB", current_rss_gb()))

    saveRDS(
        list(
            mat=mat,
            labels=labels,
            counts=counts,
            batch_levels=names(counts),
            n_cells=n_cells,
            n_vars=n_vars,
            n_hvgs=n_hvgs,
            hvg_path=hvg_path,
            h5ad_path=h5ad_path,
            cache_version=1L
        ),
        preprocess_cache,
        compress=FALSE
    )
    log_message("saved non-backend preprocessing cache: ", preprocess_cache,
        "; RSS=", sprintf("%.2f GiB", current_rss_gb()))
}

log_message("cells=", format(n_cells, big.mark=","), "; vars=", format(n_vars, big.mark=","),
    "; HVGs=", format(n_hvgs, big.mark=","), "; batches=", length(counts))
log_message("largest batches: ", paste(head(paste0(names(counts), "=", as.integer(counts)), 8L), collapse=", "))

bpparam <- BiocParallel::MulticoreParam(workers=workers, progressbar=FALSE)
bnparam <- make_bnparam(method_id)

set.seed(101L)
log_message("running fastMNN with ", method_label, " and ", workers, " workers")
timing <- system.time({
    corrected <- batchelor::fastMNN(
        mat,
        batch=batch,
        d=50L,
        k=20L,
        cos.norm=TRUE,
        correct.all=FALSE,
        BSPARAM=BiocSingular::IrlbaParam(deferred=TRUE),
        BNPARAM=bnparam,
        BPPARAM=bpparam
    )
})
log_message("fastMNN elapsed seconds=", unname(timing[["elapsed"]]),
    "; RSS=", sprintf("%.2f GiB", current_rss_gb()))

corrected_coords <- SingleCellExperiment::reducedDim(corrected, "corrected")
corrected_labels <- as.character(corrected$batch)
saveRDS(
    list(
        corrected=corrected_coords,
        dataset=labels,
        batch=corrected_labels,
        workers=workers,
        method=paste0("fastMNN + ", method_label),
        hvg_path=hvg_path
    ),
    corrected_rds,
    compress=FALSE
)
saveRDS(S4Vectors::metadata(corrected)$merge.info, merge_info_rds, compress=FALSE)
log_message("saved corrected coordinates: ", corrected_rds)
log_message("saved merge info: ", merge_info_rds)

rm(corrected, mat)
invisible(gc())

set.seed(103L)
log_message("running UMAP on corrected coordinates")
umap_coords <- uwot::umap(
    corrected_coords,
    n_neighbors=30L,
    min_dist=0.3,
    metric="euclidean",
    n_threads=workers,
    n_sgd_threads=workers,
    verbose=TRUE,
    ret_model=FALSE
)
write_umap_csv(umap_coords, labels, umap_csv)
log_message("wrote corrected UMAP coordinates: ", umap_csv)
