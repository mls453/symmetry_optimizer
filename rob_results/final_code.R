library(lmtest)
library(imager)

image_test <- function(im1, im2) {
  
  # Image 1
  dm <- dim(im1)
  n <- prod(dm[1], dm[2])
  
  colours <- matrix(c(as.vector(im1[,,,1]), rep(0, dm[1]*dm[2]*2), rep(0, dm[1]*dm[2]), as.vector(im1[,,,2]), rep(0, dm[1]*dm[2]), rep(0, dm[1]*dm[2]*2), as.vector(im1[,,,3])), ncol = 3)
  
  xy <- expand.grid(1:dm[1], 1:dm[2])
  xy <- rbind(xy, xy, xy)
  
  X1 <- cbind(xy, colours)
  names(X1) <- c("lat", "lon", "r", "g", "b")
  
  
  # Image 2
  dm <- dim(im2)
  n <- prod(dm[1], dm[2])
  
  colours <- matrix(c(as.vector(im2[,,,1]), rep(0, dm[1]*dm[2]*2), rep(0, dm[1]*dm[2]), as.vector(im2[,,,2]), rep(0, dm[1]*dm[2]), rep(0, dm[1]*dm[2]*2), as.vector(im2[,,,3])), ncol = 3)
  
  xy <- expand.grid(1:dm[1], 1:dm[2])
  xy <- rbind(xy, xy, xy)
  
  X2 <- cbind(xy, colours)
  names(X2) <- c("lat", "lon", "r", "g", "b")
  
  
  # Test
  X <- as.data.frame(rbind(X1, X2))
  y <- as.factor(c(rep(0, nrow(X1)), rep(1, nrow(X2))))
  data <- as.data.frame(cbind(y, X))
  data$lat <- as.numeric(data$lat)
  data$lon <- as.numeric(data$lon)
  fit <- glm(y ~ lat:lon:r + lat:lon:g + lat:lon:b, data = data, family = binomial())
  fit_h0 <- glm(y ~ 1, data = data, family = binomial())
  
  test <- lrtest(fit, fit_h0)$`Pr(>Chisq)`[2]
  
  return(test)
  
} 

path <- "/Users/robertomolinari/OneDrive - Auburn University/Michael/downsampled"

path <- "C:/Users/rcm0075/OneDrive - Auburn University/Michael/images"

images <- list.files(path)

result_mat <- matrix(NA, length(images), length(images))
rownames(result_mat) <- colnames(result_mat) <- as.character(1:length(images))

for(i in 1:length(images)) {
  
  im1 <- load.image(paste0(path, "/", images[i]))
  
  rownames(result_mat)[i] <- images[i]
  
  for(j in i:length(images)) {
    
    im2 <- load.image(paste0(path, "/", images[j]))
    
    colnames(result_mat)[j] <- images[j]
    
    test <- image_test(im1, im2)
    
    result_mat[i, j] <- result_mat[j, i] <- test
    
  }
  
}

# Load results
load("glm_results.Rda")
load("full_results.Rda")

ind <- which(lower.tri(result_mat, diag=F), arr.ind = TRUE)
results <- data.frame(rownames(result_mat)[ind[, 1]], colnames(result_mat)[ind[, 2]], result_mat[lower.tri(result_mat)])

# Check similarities of non-identical images
results[(results[, 3] >= 0.05) & (results[, 1] != results[, 2]), ]

# Check images known to be similar
similar <- results[(results[, 3] >= 0.05), ]
non_similar <- results[(results[, 3] < 0.05), ]

# Compare proportion of expected similarities with proportion of expected non-similarities
sum(substr(similar[,1], 1, nchar(similar[,1])-6) == substr(similar[,2], 1, nchar(similar[,2])-6))
nrow(similar)
sum(substr(non_similar[,1], 1, nchar(non_similar[,1])-6) == substr(non_similar[,2], 1, nchar(non_similar[,2])-6))
nrow(non_similar)
prop.test(c(30, 214), c(44, 10664), alternative = "greater")
fisher.test(matrix(c(30, 214-30, 44, 10664-44), ncol=2), alternative = "greater")


### Only off-diagonal elements

# Check images known to be similar
delta <- row(result_mat) - col(result_mat)
off_diag_results <- data.frame(rownames(result_mat)[2:nrow(result_mat)], rownames(result_mat)[1:(nrow(result_mat)-1)], as.numeric(result_mat[delta == 1]))
similar <- off_diag_results[(off_diag_results[, 3] >= 0.05), ]
non_similar <- off_diag_results[(off_diag_results[, 3] < 0.05), ]

# Compare proportion of expected similarities with proportion of expected non-similarities
sum(substr(similar[,1], 1, nchar(similar[,1])-6) == substr(similar[,2], 1, nchar(similar[,2])-6))
nrow(similar)
sum(substr(non_similar[,1], 1, nchar(non_similar[,1])-6) == substr(non_similar[,2], 1, nchar(non_similar[,2])-6))
nrow(non_similar)
prop.test(c(15, 9), c(18, 29), alternative = "greater")
fisher.test(matrix(c(15, 18-15, 9, 29-9), ncol=2), alternative = "greater")


######
# Analysis of non-similar but expected to be similar
non_similar[substr(non_similar[,1], 1, nchar(non_similar[,1])-6) == substr(non_similar[,2], 1, nchar(non_similar[,2])-6), ]

path <- "/Users/robertomolinari/OneDrive - Auburn University/Michael/downsampled"

path <- "C:/Users/rcm0075/OneDrive - Auburn University/Michael/downsampled"

im1 <- load.image(paste0(path, "/", "East_day169_18Jun2013_S.png"))
im2 <- load.image(paste0(path, "/", "East_day169_18Jun2013_N.png"))

im1 <- load.image(paste0(path, "/", "East_day105_16Apr2013_S.png"))
im2 <- load.image(paste0(path, "/", "East_day105_16Apr2013_N.png"))

im_diff <- im1
im_diff[,,,1] <-  1 - abs((im1[,,,1] - im2[,,,1]))
im_diff[,,,2] <-  1 - abs((im1[,,,2] - im2[,,,2]))
im_diff[,,,3] <-  1 - abs((im1[,,,3] - im2[,,,3]))

par(mfrow = c(1, 3))
plot(im1)
plot(im2)
plot(im_diff)


colour_isolate <- function(im1, im2) {
  
  colours_im1 <- matrix(c(as.vector(im1[,,,1]), as.vector(im1[,,,2]), as.vector(im1[,,,3])), ncol = 3)
  colours_im2 <- matrix(c(as.vector(im2[,,,1]), as.vector(im2[,,,2]), as.vector(im2[,,,3])), ncol = 3)
  
  total_col_im1 <- unique(colours_im1)
  total_col_im2 <- unique(colours_im2)
  total_col <- rbind(total_col_im1, total_col_im2)
  total_col <- total_col[duplicated(total_col),]
  total_col <- total_col[-1, ]
  
  test <- matrix(NA, nrow(total_col), 4)
  
  for(i in 1:nrow(total_col)) {
    
    # First image
    im1_col <- colours_im1
    im1_col[1:nrow(im1_col), 1:ncol(im1_col)] <- 1
    
    im1_col[colours_im1[, 1] == total_col[i, 1] & colours_im1[, 2] == total_col[i, 2] & colours_im1[, 3] == total_col[i, 3], ] <-  t(total_col[i, ])
    
    im1_recon <- im1
    im1_recon[, , , 1] <- matrix(im1_col[, 1], ncol = ncol(im1))
    im1_recon[, , , 2] <- matrix(im1_col[, 2], ncol = ncol(im1))
    im1_recon[, , , 3] <- matrix(im1_col[, 3], ncol = ncol(im1))
    
    # Second image
    im2_col <- colours_im2
    im2_col[1:nrow(im2_col), 1:ncol(im2_col)] <- 1
    
    im2_col[colours_im2[, 1] == total_col[i, 1] & colours_im2[, 2] == total_col[i, 2] & colours_im2[, 3] == total_col[i, 3], ] <-  t(total_col[i, ])
    
    im2_recon <- im2
    im2_recon[, , , 1] <- matrix(im2_col[, 1], ncol = ncol(im2))
    im2_recon[, , , 2] <- matrix(im2_col[, 2], ncol = ncol(im2))
    im2_recon[, , , 3] <- matrix(im2_col[, 3], ncol = ncol(im2))
    
    test[i, ] <- c(image_test(im1_recon, im2_recon), total_col[i, ])
    
  }
  
  return(test)
  
}


colour_isolate(im1, im2)


par(mfrow = c(2, 2))
plot(im1)
plot(im2)
plot(im1_recon)
plot(im2_recon)
