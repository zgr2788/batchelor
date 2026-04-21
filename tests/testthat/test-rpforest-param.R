# This file is part of the batchelor smoke tests for new BiocNeighbors backends.

library(BiocSingular)

test_that("fastMNN accepts RpforestParam", {
    skip_if_not_installed("rnndescent")
    set.seed(1200008)

    B1 <- matrix(rnorm(10000, 0), nrow=100)
    B2 <- matrix(rnorm(20000, 1), nrow=100)
    param <- BiocNeighbors::RpforestParam()

    out <- fastMNN(B1, B2, d=20, BSPARAM=ExactParam(), BNPARAM=param)
    expect_identical(dim(reducedDim(out)), c(ncol(B1) + ncol(B2), 20L))
    expect_identical(as.integer(out$batch), rep(1:2, c(ncol(B1), ncol(B2))))
    expect_true(nrow(metadata(out)$merge.info$pairs[[1]]) > 0)
})
