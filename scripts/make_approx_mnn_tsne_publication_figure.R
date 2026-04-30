#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(ggplot2)
    library(Rtsne)
})

repo_root <- "/insomnia001/depts/morpheus/users/ob2391/Rstudio/batchelor"
cache_dir <- file.path(repo_root, "vignettes", "approx-mnn_cache", "html")
figure_dir <- file.path(repo_root, "figures")
dir.create(figure_dir, recursive=TRUE, showWarnings=FALSE)

prepare_cache <- file.path(
    cache_dir,
    "prepare-pcs_fbb9884ec6352523a25b16823581a4f3"
)
corrected_cache <- file.path(
    cache_dir,
    "corrected-tsne-data_49fe619317b3aaeaa45c8b1cbfbc37df"
)

required_cache_files <- c(
    paste0(prepare_cache, ".rdb"),
    paste0(prepare_cache, ".rdx"),
    paste0(corrected_cache, ".rdb"),
    paste0(corrected_cache, ".rdx")
)
missing <- required_cache_files[!file.exists(required_cache_files)]
if (length(missing)) {
    stop("Missing required vignette cache files:\n", paste(missing, collapse="\n"))
}

lazyLoad(corrected_cache)
if (!exists("tsne_comparison")) {
    stop("Corrected t-SNE cache did not contain `tsne_comparison`.")
}
if (!exists("tsne_seed")) {
    tsne_seed <- 103L
}

corrected <- tsne_comparison[
    as.character(tsne_comparison$method) %in% c("Kmknn (exact)", "FLANN k-means"),
    c("TSNE1", "TSNE2", "batch", "method")
]
corrected$panel <- ifelse(
    as.character(corrected$method) == "Kmknn (exact)",
    "B. Exact fastMNN",
    "C. FLANN k-means fastMNN"
)
corrected <- corrected[, c("TSNE1", "TSNE2", "batch", "panel")]

lazyLoad(prepare_cache)
if (!exists("pcs")) {
    stop("Shared PCA cache did not contain `pcs`.")
}

pcs <- as.list(pcs)
uncorrected_input <- do.call(rbind, pcs)
uncorrected_batch <- factor(
    rep(c("Zeisel", "Tasic"), vapply(pcs, nrow, integer(1))),
    levels=c("Zeisel", "Tasic")
)

set.seed(tsne_seed)
uncorrected_coords <- Rtsne::Rtsne(
    uncorrected_input,
    pca=FALSE,
    check_duplicates=FALSE
)$Y

uncorrected <- data.frame(
    TSNE1=uncorrected_coords[, 1],
    TSNE2=uncorrected_coords[, 2],
    batch=uncorrected_batch,
    panel="A. Uncorrected",
    stringsAsFactors=FALSE
)

plot_data <- rbind(uncorrected, corrected)
plot_data$batch <- factor(plot_data$batch, levels=c("Zeisel", "Tasic"))
plot_data$panel <- factor(
    plot_data$panel,
    levels=c("A. Uncorrected", "B. Exact fastMNN", "C. FLANN k-means fastMNN")
)

normalize_panel <- function(x) {
    x$TSNE1 <- x$TSNE1 - median(x$TSNE1)
    x$TSNE2 <- x$TSNE2 - median(x$TSNE2)
    scale <- max(diff(range(x$TSNE1)), diff(range(x$TSNE2)))
    x$TSNE1 <- x$TSNE1 / scale
    x$TSNE2 <- x$TSNE2 / scale
    x
}
plot_data <- do.call(rbind, lapply(split(plot_data, plot_data$panel), normalize_panel))
plot_data$panel <- factor(
    plot_data$panel,
    levels=c("A. Uncorrected", "B. Exact fastMNN", "C. FLANN k-means fastMNN")
)

palette <- c(Zeisel="#0072B2", Tasic="#D55E00")

tsne_plot <- ggplot(plot_data, aes(TSNE1, TSNE2, colour=batch)) +
    geom_point(size=0.38, alpha=0.72, stroke=0) +
    facet_wrap(~ panel, nrow=1) +
    coord_equal() +
    scale_colour_manual(values=palette) +
    labs(x="t-SNE 1", y="t-SNE 2", colour="Dataset") +
    guides(colour=guide_legend(override.aes=list(size=2.5, alpha=1))) +
    theme_classic(base_size=10) +
    theme(
        strip.background=element_blank(),
        strip.text=element_text(face="bold", size=10.5, margin=margin(b=5)),
        axis.text=element_blank(),
        axis.ticks=element_blank(),
        axis.line=element_line(linewidth=0.35, colour="black"),
        axis.title=element_text(size=9),
        legend.position="bottom",
        legend.title=element_text(size=9),
        legend.text=element_text(size=9),
        legend.key.width=unit(0.35, "in"),
        panel.spacing=unit(0.45, "lines"),
        plot.margin=margin(5.5, 7, 4, 5.5)
    )

output <- file.path(figure_dir, "approx_mnn_tsne_uncorrected_exact_flann_kmeans.pdf")
ggsave(
    filename=output,
    plot=tsne_plot,
    device="pdf",
    width=7.2,
    height=2.8,
    units="in",
    useDingbats=FALSE,
    bg="white"
)

message("Wrote ", output, " (", file.info(output)$size, " bytes)")
