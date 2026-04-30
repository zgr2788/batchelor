#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(batchelor)
    library(BiocNeighbors)
    library(BiocParallel)
    library(cluster)
    library(ggplot2)
})

parse_arg <- function(name, default) {
    prefix <- paste0("--", name, "=")
    hit <- commandArgs(trailingOnly=TRUE)
    hit <- hit[startsWith(hit, prefix)]
    if (length(hit)) {
        sub(prefix, "", hit[[1]], fixed=TRUE)
    } else {
        default
    }
}

script_dir <- function() {
    file_arg <- commandArgs(FALSE)
    file_arg <- file_arg[startsWith(file_arg, "--file=")]
    if (length(file_arg)) {
        return(dirname(normalizePath(sub("^--file=", "", file_arg[[1]]))))
    }
    getwd()
}

repo_root <- normalizePath(file.path(script_dir(), ".."))
cache_dir <- file.path(repo_root, "vignettes", "approx-mnn_cache", "html")
figure_dir <- file.path(repo_root, "figures")
dir.create(figure_dir, recursive=TRUE, showWarnings=FALSE)

workers <- as.integer(parse_arg("workers", "32"))
nboot <- as.integer(parse_arg("nboot", "5000"))
set.seed(as.integer(parse_arg("seed", "101")))

prepare_cache <- sub(
    "\\.rdb$", "",
    list.files(cache_dir, pattern="^prepare-pcs_.*\\.rdb$", full.names=TRUE)
)
if (length(prepare_cache) != 1L) {
    stop("Expected exactly one prepare-pcs cache, found ", length(prepare_cache), ".")
}

cache_env <- new.env(parent=emptyenv())
lazyLoad(prepare_cache, envir=cache_env)
if (!exists("pcs", envir=cache_env)) {
    stop("The prepare-pcs cache did not contain `pcs`.")
}

pcs <- as.list(get("pcs", envir=cache_env))
if (length(pcs) != 2L) {
    stop("Expected exactly two PCA batches, found ", length(pcs), ".")
}

batch_levels <- c("Zeisel", "Tasic")
batch <- factor(
    rep(batch_levels, vapply(pcs, nrow, integer(1))),
    levels=batch_levels
)

bpparam <- MulticoreParam(workers=workers)

message("Loaded shared PCA: ", paste(vapply(pcs, nrow, integer(1)), collapse=" + "),
    " cells x ", ncol(pcs[[1]]), " dimensions")

embedding_specs <- list(
    uncorrected=list(
        label="Uncorrected shared PCA",
        short_label="Uncorrected",
        matrix=do.call(rbind, pcs)
    ),
    exact=list(
        label="Exact fastMNN",
        short_label="Exact",
        param=KmknnParam()
    ),
    flann_kmeans=list(
        label="FLANN k-means fastMNN",
        short_label="FLANN k-means",
        param=FlannKmeansParam()
    )
)

for (method in setdiff(names(embedding_specs), "uncorrected")) {
    message("Running reducedMNN for ", embedding_specs[[method]]$label)
    set.seed(101)
    corrected <- reducedMNN(
        pcs[[1]],
        pcs[[2]],
        BNPARAM=embedding_specs[[method]]$param,
        BPPARAM=bpparam
    )
    embedding_specs[[method]]$matrix <- as.matrix(corrected$corrected)
}

bootstrap_ci <- function(x, nboot, seed) {
    if (nboot <= 0L) {
        return(c(low=NA_real_, high=NA_real_))
    }
    set.seed(seed)
    means <- replicate(nboot, mean(sample(x, length(x), replace=TRUE)))
    stats::quantile(means, c(0.025, 0.975), names=FALSE)
}

per_cell <- do.call(rbind, lapply(seq_along(embedding_specs), function(i) {
    spec <- embedding_specs[[i]]
    method_id <- names(embedding_specs)[[i]]
    message("Computing batch silhouette widths for ", spec$label)
    sil <- cluster::silhouette(as.integer(batch), stats::dist(spec$matrix))
    data.frame(
        method_id=method_id,
        method=spec$label,
        short_method=spec$short_label,
        batch=batch,
        silhouette_width=sil[, "sil_width"],
        stringsAsFactors=FALSE
    )
}))

method_order <- vapply(embedding_specs, `[[`, character(1), "label")
short_order <- vapply(embedding_specs, `[[`, character(1), "short_label")

summary_df <- do.call(rbind, lapply(split(per_cell, per_cell$method_id), function(x) {
    ci <- bootstrap_ci(x$silhouette_width, nboot=nboot, seed=303)
    data.frame(
        method_id=x$method_id[[1]],
        method=x$method[[1]],
        short_method=x$short_method[[1]],
        n_cells=nrow(x),
        mean_silhouette=mean(x$silhouette_width),
        median_silhouette=stats::median(x$silhouette_width),
        mean_abs_silhouette=mean(abs(x$silhouette_width)),
        ci_low=ci[[1]],
        ci_high=ci[[2]],
        stringsAsFactors=FALSE
    )
}))
summary_df$method <- factor(summary_df$method, levels=method_order)
summary_df$short_method <- factor(summary_df$short_method, levels=short_order)
summary_df <- summary_df[order(summary_df$method), , drop=FALSE]

per_batch_df <- aggregate(
    silhouette_width ~ method_id + method + short_method + batch,
    per_cell,
    function(x) c(mean=mean(x), median=stats::median(x), mean_abs=mean(abs(x)))
)
per_batch_df <- do.call(data.frame, per_batch_df)
names(per_batch_df) <- sub("silhouette_width\\.", "", names(per_batch_df))

metrics_file <- file.path(figure_dir, "approx_mnn_embedding_batch_silhouette_metrics.csv")
per_batch_file <- file.path(figure_dir, "approx_mnn_embedding_batch_silhouette_by_dataset.csv")
per_cell_file <- file.path(figure_dir, "approx_mnn_embedding_batch_silhouette_per_cell.csv.gz")

utils::write.csv(summary_df, metrics_file, row.names=FALSE, quote=FALSE)
utils::write.csv(per_batch_df, per_batch_file, row.names=FALSE, quote=FALSE)
utils::write.csv(per_cell, gzfile(per_cell_file), row.names=FALSE, quote=FALSE)

palette <- c(
    "Uncorrected"="#6B7280",
    "Exact"="#0072B2",
    "FLANN k-means"="#D55E00"
)

y_range <- range(c(summary_df$ci_low, summary_df$ci_high, 0), na.rm=TRUE)
y_pad <- max(diff(y_range) * 0.18, 0.015)
summary_df$label_y <- ifelse(
    summary_df$mean_silhouette >= 0,
    summary_df$mean_silhouette + y_pad * 0.18,
    summary_df$mean_silhouette - y_pad * 0.18
)
summary_df$label_vjust <- ifelse(summary_df$mean_silhouette >= 0, 0, 1)

metric_plot <- ggplot(summary_df, aes(short_method, mean_silhouette, fill=short_method)) +
    geom_hline(yintercept=0, linewidth=0.35, linetype="dashed", colour="grey35") +
    geom_col(width=0.62, colour="black", linewidth=0.28) +
    geom_errorbar(aes(ymin=ci_low, ymax=ci_high), width=0.14, linewidth=0.35) +
    geom_text(
        aes(y=label_y, label=sprintf("%.3f", mean_silhouette), vjust=label_vjust),
        size=3.1
    ) +
    scale_fill_manual(values=palette) +
    coord_cartesian(ylim=c(y_range[[1]] - y_pad, y_range[[2]] + y_pad), clip="off") +
    labs(
        x=NULL,
        y="Batch average silhouette width"
    ) +
    theme_classic(base_size=10) +
    theme(
        legend.position="none",
        axis.text.x=element_text(size=9.5, colour="black"),
        axis.text.y=element_text(size=8.5, colour="black"),
        axis.title.y=element_text(size=9.5, margin=margin(r=7)),
        axis.line=element_line(linewidth=0.35, colour="black"),
        axis.ticks=element_line(linewidth=0.3, colour="black"),
        plot.margin=margin(6, 8, 5, 6)
    )

output <- file.path(figure_dir, "approx_mnn_embedding_batch_silhouette_comparison.pdf")
ggsave(
    filename=output,
    plot=metric_plot,
    device="pdf",
    width=4.2,
    height=3.0,
    units="in",
    useDingbats=FALSE,
    bg="white"
)

message("Wrote ", output, " (", file.info(output)$size, " bytes)")
message("Wrote ", metrics_file)
message("Wrote ", per_batch_file)
message("Wrote ", per_cell_file)
