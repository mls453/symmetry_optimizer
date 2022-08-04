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
  
  x1_col <- paste0(X1[, 3], X1[, 4], X1[, 5])
  
  
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

colour_isolate <- function(im1, im2, col = c("#000000","#0000FF", "#00AEF0", "#5C5C5C", "#D8FFC5", "#F26522", "#FFFF56", "#FFFFFF")) {
  
  colours_im1 <- rgb(as.vector(im1[,,,1]), as.vector(im1[,,,2]), as.vector(im1[,,,3]))
  colours_im2 <- rgb(as.vector(im2[,,,1]), as.vector(im2[,,,2]), as.vector(im2[,,,3]))
  
  col_freq <- rep(NA, length(col))
  
  for(i in 1:length(col)) {
    
    index1 <- ifelse(colours_im1 == col[i], 1, 0)
    index2 <- ifelse(colours_im2 == col[i], 1, 0)
    overlap <- sum(index1*index2)
    col_freq[i] <- overlap/(sum(index1) + sum(index2) - overlap)
    
  }
  
  return(col_freq)
  
}

path <- "/Users/robertomolinari/OneDrive - Auburn University/Michael/images"

path <- "C:/Users/rcm0075/OneDrive - Auburn University/Michael/images"

images <- list.files(path)
n <- length(images)*(length(images) - 1)/2

results <- data.frame("Image_1" = rep("NA", n), "Image_2" = rep("NA", n), "p_value" = rep(0, n), "Col_1" = rep(0, n), "Col_2" = rep(0, n), "Col_3" = rep(0, n), "Col_4" = rep(0, n), "Col_5" = rep(0, n), "Col_6" = rep(0, n), "Col_7" = rep(0, n), "Col_8" = rep(0, n))


pb <- txtProgressBar(min = 0,
                     max = n,
                     style = 3,
                     width = 50,
                     char = "=") 

iter <- 1

for(i in 1:length(images)) {
  
  im1 <- load.image(paste0(path, "/", images[i]))
  
  for(j in i:length(images)) {
    
    im2 <- load.image(paste0(path, "/", images[j]))
    
    results[iter, 2] <- images[j]
    results[iter, 1] <- images[i]
    
    results[iter, 3] <- image_test(im1, im2)
    
    results[iter, 4:11] <- colour_isolate(im1, im2)  
    
    iter <- iter + 1
    
    setTxtProgressBar(pb, iter)
    
  }
  
}

save(results, file = "next_symmetry_tests.Rda")

write.csv(results, file = "nest_symmetry_tests.csv")
