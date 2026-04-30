#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(ggplot2)
    library(Rtsne)
})

repo_root <- "/insomnia001/depts/morpheus/users/ob2391/Rstudio/batchelor"
figure_dir <- file.path(repo_root, "vignettes", "figures", "approx-mnn")
dir.create(figure_dir, recursive=TRUE, showWarnings=FALSE)

cache_base <- file.path(
    repo_root,
    "vignettes",
    "approx-mnn_cache",
    "html",
    "prepare-pcs_fbb9884ec6352523a25b16823581a4f3"
)

if (!file.exists(paste0(cache_base, ".rdb")) || !file.exists(paste0(cache_base, ".rdx"))) {
    stop("Could not find cached pre-correction PCA object at: ", cache_base)
}

lazyLoad(cache_base)
if (!exists("pcs")) {
    stop("The cached object did not contain `pcs`.")
}

tsne_seed <- 103L
pcs <- as.list(pcs)
uncorrected <- do.call(rbind, pcs)
batch <- factor(
    rep(c("Zeisel", "Tasic"), vapply(pcs, nrow, integer(1))),
    levels=c("Zeisel", "Tasic")
)

set.seed(tsne_seed)
coords <- Rtsne(
    uncorrected,
    pca=FALSE,
    check_duplicates=FALSE
)$Y

tsne_uncorrected <- data.frame(
    TSNE1=coords[, 1],
    TSNE2=coords[, 2],
    batch=batch
)

tsne_plot <- ggplot(tsne_uncorrected, aes(TSNE1, TSNE2, colour=batch)) +
    geom_point(size=0.5, alpha=0.6) +
    labs(
        x="t-SNE 1",
        y="t-SNE 2",
        colour="batch",
        subtitle=paste0("Example data, uncorrected shared PCA space, t-SNE seed = ", tsne_seed)
    ) +
    theme_minimal()

output <- file.path(figure_dir, "06-uncorrected-tsne.pdf")
ggsave(
    filename=output,
    plot=tsne_plot,
    device="pdf",
    width=9.5,
    height=7.5,
    units="in",
    dpi=300,
    bg="white"
)

message("Wrote ", output, " (", file.info(output)$size, " bytes)")
