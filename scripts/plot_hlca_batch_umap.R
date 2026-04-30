#!/usr/bin/env Rscript

Sys.setenv(HDF5_USE_FILE_LOCKING = "FALSE")

suppressPackageStartupMessages({
  library(anndataR)
  library(rhdf5)
  library(ggplot2)
  library(scattermore)
  library(cluster)
})

input <- "/insomnia001/depts/morpheus/users/ob2391/Rstudio/batchelor/data/hlca_core_uncompressed.h5ad"
output <- "/insomnia001/depts/morpheus/users/ob2391/Rstudio/batchelor/figures/hlca_core_umap_by_study_batch.pdf"
metrics_output <- "/insomnia001/depts/morpheus/users/ob2391/Rstudio/batchelor/figures/hlca_core_batch_silhouette_by_study.csv"

dir.create(dirname(output), showWarnings = FALSE, recursive = TRUE)

message("Opening with anndataR: ", input)
adata <- read_h5ad(input, as = "HDF5AnnData", mode = "r")
message("anndataR object class: ", paste(class(adata), collapse = ", "))
rm(adata)
gc()
h5closeAll()

message("Reading UMAP and study batch labels")
umap <- h5read(input, "/obsm/X_umap")
if (nrow(umap) == 2L) {
  umap <- t(umap)
} else if (ncol(umap) != 2L) {
  stop("Expected /obsm/X_umap to have two dimensions, got: ", paste(dim(umap), collapse = " x "))
}

categories <- h5read(input, "/obs/study/categories")
codes <- h5read(input, "/obs/study/codes")
study <- ifelse(codes < 0L, NA_character_, categories[codes + 1L])

plot_df <- data.frame(
  UMAP_1 = umap[, 1],
  UMAP_2 = umap[, 2],
  study = factor(study, levels = categories)
)

counts <- sort(table(plot_df$study), decreasing = TRUE)
study_levels <- names(counts)
plot_df$study <- factor(as.character(plot_df$study), levels = study_levels)
set.seed(1)
plot_df <- plot_df[sample.int(nrow(plot_df)), , drop = FALSE]

palette <- c(
  "#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#A65628",
  "#F781BF", "#1B9E77", "#D95F02", "#7570B3", "#66A61E"
)[seq_along(study_levels)]
names(palette) <- study_levels
legend_labels <- paste0(study_levels, " (n=", format(as.integer(counts), big.mark = ","), ")")
names(legend_labels) <- study_levels

message("Calculating UMAP silhouette by study batch")
silhouette_per_batch <- 1000L
silhouette_indices <- unlist(
  lapply(study_levels, function(level) {
    indices <- which(plot_df$study == level)
    sample(indices, min(length(indices), silhouette_per_batch))
  }),
  use.names = FALSE
)
silhouette_df <- plot_df[silhouette_indices, , drop = FALSE]
silhouette_dist <- dist(as.matrix(silhouette_df[, c("UMAP_1", "UMAP_2")]))
silhouette_fit <- silhouette(as.integer(silhouette_df$study), silhouette_dist)
silhouette_values <- silhouette_fit[, "sil_width"]
silhouette_summary <- aggregate(
  silhouette_width ~ study,
  data = data.frame(study = silhouette_df$study, silhouette_width = silhouette_values),
  FUN = mean
)
silhouette_summary$cells_in_batch <- as.integer(counts[as.character(silhouette_summary$study)])
silhouette_summary$cells_sampled <- as.integer(table(silhouette_df$study)[as.character(silhouette_summary$study)])
silhouette_summary <- silhouette_summary[order(silhouette_summary$silhouette_width, decreasing = TRUE), ]
overall_silhouette <- mean(silhouette_values)

write.csv(silhouette_summary, metrics_output, row.names = FALSE)
message(
  "Mean UMAP silhouette by study batch: ",
  round(overall_silhouette, 4),
  " (stratified n=",
  length(silhouette_indices),
  ")"
)
message("Wrote metrics: ", metrics_output)

theme_umap <- theme_classic(base_size = 10) +
  theme(
    axis.title = element_text(size = 9),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    legend.title = element_text(size = 9),
    legend.text = element_text(size = 7),
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(size = 9),
    strip.text = element_text(size = 7)
  )

p_all <- ggplot(plot_df, aes(UMAP_1, UMAP_2, color = study)) +
  geom_scattermore(pointsize = 2, pixels = c(2600, 2050), alpha = 0.9) +
  coord_equal() +
  scale_color_manual(values = palette, labels = legend_labels, drop = FALSE) +
  labs(
    title = "HLCA core UMAP colored by study batch",
    subtitle = paste0(
      format(nrow(plot_df), big.mark = ","),
      " cells; mean UMAP silhouette by study = ",
      sprintf("%.3f", overall_silhouette),
      " (stratified n=",
      format(length(silhouette_indices), big.mark = ","),
      ")"
    ),
    x = "UMAP 1",
    y = "UMAP 2",
    color = "Study batch"
  ) +
  guides(color = guide_legend(override.aes = list(size = 4, alpha = 1))) +
  theme_umap

p_facets <- ggplot(plot_df, aes(UMAP_1, UMAP_2, color = study)) +
  geom_scattermore(pointsize = 2, pixels = c(850, 650), alpha = 0.95) +
  coord_equal() +
  facet_wrap(~study, ncol = 4) +
  scale_color_manual(values = palette, drop = FALSE) +
  labs(
    title = "HLCA core UMAP split by study batch",
    x = "UMAP 1",
    y = "UMAP 2"
  ) +
  theme_umap +
  theme(legend.position = "none")

count_df <- data.frame(
  study = factor(study_levels, levels = rev(study_levels)),
  cells = as.integer(counts)
)

p_counts <- ggplot(count_df, aes(cells, study, fill = study)) +
  geom_col(width = 0.75) +
  scale_fill_manual(values = palette, drop = FALSE) +
  scale_x_continuous(labels = function(x) format(x, big.mark = ",")) +
  labs(
    title = "Cell counts by study batch",
    x = "Cells",
    y = NULL
  ) +
  theme_classic(base_size = 10) +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", size = 12),
    axis.text.y = element_text(size = 8)
  )

p_silhouette <- ggplot(
  silhouette_summary,
  aes(silhouette_width, reorder(study, silhouette_width), fill = study)
) +
  geom_vline(xintercept = 0, color = "grey55", linewidth = 0.35) +
  geom_col(width = 0.75) +
  scale_fill_manual(values = palette, drop = FALSE) +
  labs(
    title = "Study-batch separability on current UMAP",
    subtitle = paste0(
      "Mean silhouette = ",
      sprintf("%.3f", overall_silhouette),
      "; Euclidean UMAP distances; stratified sample up to ",
      format(silhouette_per_batch, big.mark = ","),
      " cells per study"
    ),
    x = "Mean silhouette width",
    y = NULL
  ) +
  theme_classic(base_size = 10) +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(size = 9),
    axis.text.y = element_text(size = 8)
  )

message("Writing: ", output)
pdf(output, width = 11, height = 8.5, useDingbats = FALSE)
print(p_all)
print(p_facets)
print(p_counts)
print(p_silhouette)
dev.off()

message("Wrote ", output, " (", format(file.info(output)$size, big.mark = ","), " bytes)")
