library(corrplot)
library(ggplot2)
library(MASS)
library(glmnet)
library(randomForest)
library(e1071)
library(ggcorrplot)
library(xgboost)
library(splines)
library(mgcv)

load("./data/Data.train.update.RData")
load("./data/Data.test.update.RData")

head(Data.train)
head(Data.test)

dim(Data.train)
dim(Data.test)

#### ---- EDA ---- ####
str(Data.train)

# Check for missing values
colSums(is.na(Data.train))

# Univariate Analysis #
summary(Data.train)

# Histograms for top varying columns
var_values <- apply(Data.train[, 1:140], 2, var)
top_vars <- names(sort(var_values, decreasing = TRUE)[1:9])
par(mfrow = c(3, 3)); for (col in top_vars) {
  hist(Data.train[[col]], main = col, xlab = col)
}

# Q-Q plots for top varying columns
par(mfrow = c(3, 3)); for (col in top_vars) {
  qqnorm(Data.train[[col]], main = col)
  qqline(Data.train[[col]], col = "steelblue", lwd = 2)
}

# Bivariate Analysis #
# par(mfrow = c(1, 1))
ggcorrplot(cor(Data.train, use = "complete.obs"))

# Identify top 6 predictors by absolute correlation with Y
top5 <- names(sort(abs(cor(Data.train)[,"Y"]), decreasing=TRUE))[2:7]

# Plot scatter plots with lowess smoother for each top predictor
par(mfrow=c(2,3)); for (i in top5) {
  plot(Data.train[[i]], Data.train$Y,
       main=paste("Y vs", i),
       xlab=i, ylab="Y", pch=16, col=rgb(0, 0, 0, 0.3))
  lines(lowess(Data.train[[i]], Data.train$Y), col="red", lwd=2)
}

#### ---- Preprocessing ---- ####

# Function for scaling
scaling <- function(data, mode, means=NULL, sds=NULL) {
  
  if (mode == 'train') {
    
    # Exclude Y
    X <- data[, -which(names(data) == "Y")]
    means <- apply(X, 2, mean)
    sds   <- apply(X, 2, sd)
    
    # Scale train and test
    processed_data <- as.data.frame(scale(X, center = means, scale = sds))
    
    # Add Y back
    processed_data$Y <- data$Y
    
    return (list(
      processed_data = processed_data,
      means = means,
      sds = sds
    ))
  }
  
  else {
    X <- data[, -which(names(data) == "Y")]
    processed_data  <- as.data.frame(scale(X, center = means, scale = sds))
    # Add Y back
    processed_data$Y <- data$Y
    
    return (processed_data)
  }
    
}

# Train - Validation Split
set.seed(123)
n        <- nrow(Data.train)
train_ix <- sample(n, size = 0.8 * n)
tr_split  <- Data.train[train_ix, ]
vl_split  <- Data.train[-train_ix, ]

# Scaling
scaled_tr_data <- scaling(tr_split, mode='train')
scaled_tr = scaled_tr_data$processed_data
means <- scaled_tr_data$means
sds <- scaled_tr_data$sds
scaled_vl <- scaling(vl_split, mode="val", means = means, sds = sds)

#### ---- Base Model ---- ####
## Linear Regression
base_lm <- lm(Y ~ ., data = scaled_tr)
summary(base_lm)     # coefficients, R², p‑values
par(mfrow = c(2,2)); plot(base_lm)

train_mse <- mean(residuals(base_lm)^2)
valid_preds <- predict(base_lm, newdata = scaled_vl)
valid_mse <- mean((valid_preds - scaled_vl$Y)^2)

print(train_mse)
cat('MSE:', valid_mse, 'RMSE:', sqrt(valid_mse))

# pca_result <- prcomp(processed_train[, -which(names(processed_train) == "Y")], center = FALSE, scale. = FALSE)
# summary(pca_result)
# sd.v <- pca_result$sdev
# plot(log10(sd.v), main="Standard Deviations of Principal Components")

## RandomForest
rf <- randomForest(Y ~ ., data = scaled_tr, ntree = 500, mtry = sqrt(ncol(scaled_tr)))
pred_rf <- predict(rf, scaled_vl)
mse_rf  <- mean((scaled_vl$Y - pred_rf)^2)
cat("Random Forest MSE:", round(mse_rf,1), "RMSE", sqrt(mse_rf))

## SVM
svm_rbf <- svm(Y ~ ., data=scaled_tr,
               type = "eps-regression", kernel = "radial",
               cost = 1, gamma = 0.01, epsilon = 0.1)
pred_svr <- predict(svm_rbf, scaled_vl[, -which(names(scaled_vl) == "Y")])
mse_svr  <- mean((scaled_vl$Y - pred_svr)^2)
cat("SVR (RBF) MSE:", round(mse_svr,1), "\n")

## Polynomial Regression
# Degree of polynomial
deg <- 3 

# Predictor names (exclude Y)
predictors <- setdiff(names(tr_split), "Y")

# Build polynomial terms
poly_terms <- sapply(predictors, function(p) paste0("poly(", p, ", ", deg, ", raw = FALSE)"))

# Full formula                           
fmla <- as.formula(paste("Y ~", paste(poly_terms, collapse = " + ")))

# 5. Fit model                            
poly_mod <- lm(fmla, data = tr_split)
summary(poly_mod)
par(mfrow = c(2, 2)); plot(poly_mod)

pred_val <- predict(poly_mod, newdata = vl_split)
mse_poly <- mean((vl_split$Y - pred_val)^2)
cat("Polynomial ( deg =", deg, ")")
cat('MSE:', mse_poly, 'RMSE:', sqrt(mse_poly))

## Spline
common_df <- 3
spline_terms <- paste0("bs(", predictors, ", df=", common_df, ")")

# 4. Collapse into one big formula
fmla_bs_all <- as.formula(
  paste("Y ~", paste(spline_terms, collapse = " + "))
)

# 5. Fit and evaluate
mod_bs_all  <- lm(fmla_bs_all, data = tr_split)
summary(mod_bs_all)
par(mfrow = c(2, 2)); plot(mod_bs_all)
pred_bs_all <- predict(mod_bs_all, newdata = vl_split)
mse_bs_all  <- mean((vl_split$Y - pred_bs_all)^2)
cat("MSE (all splines):", mse_bs_all, "\n")

## GAM ##

# Step 1: Perform PCA on the training data
pca_res <- prcomp(scaled_tr[, predictors], scale. = FALSE)  # Data is already scaled

# Step 2: Determine the number of PCs to retain (explaining 95% variance)
var_explained <- cumsum(pca_res$sdev^2) / sum(pca_res$sdev^2)
k <- which(var_explained >= 0.95)[1]

# Step 3: Create a new training dataset with the first k PCs and Y
train_pca <- as.data.frame(pca_res$x[, 1:k])
colnames(train_pca) <- paste0("PC", 1:k)
train_pca$Y <- scaled_tr$Y

# Step 4: Build the GAM formula using the PCs
gam_fmla_pca <- as.formula(paste0("Y ~ ", paste0("s(PC", 1:k, ")", collapse = " + ")))

# Step 5: Fit the GAM model
gam_mod_pca <- gam(gam_fmla_pca, data = train_pca, method = "REML")

# Step 6: Transform the validation data using the same PCA
valid_pca <- predict(pca_res, newdata = scaled_vl[, predictors])
valid_pca <- as.data.frame(valid_pca[, 1:k])
colnames(valid_pca) <- paste0("PC", 1:k)

# Step 7: Predict on the validation set
pred_gam_pca <- predict(gam_mod_pca, newdata = valid_pca)

# Step 8: Compute MSE and RMSE
mse_gam_pca <- mean((scaled_vl$Y - pred_gam_pca)^2)
rmse_gam_pca <- sqrt(mse_gam_pca)
cat("GAM with PCA MSE:", round(mse_gam_pca, 4), "   RMSE:", round(rmse_gam_pca, 4), "\n")

#### ---- Cross Validation ---- ####
# Polynomial
CV.poly.f <- function(data, preds, response, degrees, k=10, Rounds=5) {
  n      <- nrow(data)
  m      <- length(degrees)
  MSEs   <- matrix(NA, nrow = m, ncol = Rounds,
                   dimnames = list(paste0("deg", degrees), paste0("R",1:Rounds)))
  
  set.seed(100)
  for (r in seq_len(Rounds)) {
    # create random folds
    fold_ids <- sample(rep(1:k, length.out = n))
    
    for (d_i in seq_along(degrees)) {
      deg <- degrees[d_i]
      mses <- numeric(k)
      
      for (fold in 1:k) {
        train_idx <- which(fold_ids != fold)
        test_idx  <- which(fold_ids == fold)
        train_df  <- data[train_idx, , drop=FALSE]
        test_df   <- data[test_idx,  , drop=FALSE]
        
        # build formula: Y ~ poly(X1,deg,raw=TRUE) + poly(X2,deg,raw=TRUE) + ...
        poly_terms <- paste0("poly(", preds, ", ", deg, ", raw=TRUE)", collapse = " + ")
        fmla <- as.formula(paste(response, "~", poly_terms))
        
        # fit and predict
        mod  <- lm(fmla, data = train_df)
        yhat <- predict(mod, newdata = test_df)
        
        # MSE for this fold
        y_true <- test_df[[response]]
        mses[fold] <- mean((y_true - yhat)^2)
      }
      
      MSEs[d_i, r] <- mean(mses)
    }
  }
  
  return(MSEs)
}

# Plot.PolyCV: plot mean ± 2 × SE of MSE vs. degree
#   MSEs    : output matrix from CV.poly.f
#   degrees : degrees vector corresponding to rows of MSEs
Plot.PolyCV <- function(MSEs, degrees) {
  Rounds <- ncol(MSEs)
  mean_mse <- rowMeans(MSEs)
  se_mse   <- apply(MSEs, 1, sd) / sqrt(Rounds)
  
  lower <- mean_mse - 2 * se_mse
  upper <- mean_mse + 2 * se_mse
  
  mat <- cbind(mean_mse, lower, upper)
  
  matplot(degrees, mat, type = "b", pch = c(19,NA,NA), lty = c(1,2,2),
          col = c("black","gray","gray"),
          xlab = "Polynomial Degree",
          ylab = "CV MSE",
          main = "10-fold CV: MSE vs. Polynomial Degree")
  legend("topright", legend = c("Mean MSE","±2 × SE"),
         lty = c(1,2), col = c("black","gray"), bty="n")
}

degrees <- 1:5     # try degrees 1 through 5
k       <- 10      # 10‑fold CV
Rounds  <- 3       # 3 repeats

# 2. Run CV
predictors <- setdiff(names(Data.train), "Y")
mse_mat <- CV.poly.f(data = Data.train,
                     preds = predictors,
                     response = "Y",
                     degrees = degrees,
                     k = k,
                     Rounds = Rounds)

# 3. Inspect raw MSE matrix
print(mse_mat)

# 4. Plot mean and error bars
Plot.PolyCV(mse_mat, degrees)

# 5. Choose optimal degree
mean_mse <- rowMeans(mse_mat)
optimal  <- degrees[which.min(mean_mse)]
print("Mean MSE:")
print(mean_mse)
cat("Optimal polynomial degree:", optimal, "\n")

# Spline
CV.spline.f <- function(data, preds, response,
                        dfs,      # vector of df values to try, e.g. 2:8
                        k = 10,   # number of folds
                        Rounds = 5,  # repeat CV this many times
                        seed = 100) {
  
  n   <- nrow(data)
  m   <- length(dfs)
  MSEs <- matrix(NA, nrow = m, ncol = Rounds,
                 dimnames = list(paste0("df", dfs),
                                 paste0("R", seq_len(Rounds))))
  
  set.seed(seed)
  for (r in seq_len(Rounds)) {
    # random folds (balanced)
    fold_ids <- sample(rep(seq_len(k), length.out = n))
    
    for (i in seq_along(dfs)) {
      df_i <- dfs[i]
      fold_mses <- numeric(k)
      
      for (fold in seq_len(k)) {
        ## Split
        train_idx <- which(fold_ids != fold)
        test_idx  <- which(fold_ids == fold)
        tr_df <- data[train_idx, , drop = FALSE]
        te_df <- data[test_idx,  , drop = FALSE]
        
        ## Build spline formula: bs(Xj, df=df_i) for each predictor
        spline_terms <- paste0("bs(", preds, ", df=", df_i, ")")
        fmla <- as.formula(paste(response, "~",
                                 paste(spline_terms, collapse = " + ")))
        
        ## Fit & predict
        mod   <- lm(fmla, data = tr_df)
        yhat  <- predict(mod, newdata = te_df)
        
        ## Compute MSE
        y_true <- te_df[[response]]
        fold_mses[fold] <- mean((y_true - yhat)^2)
      }
      
      MSEs[i, r] <- mean(fold_mses)
    }
  }
  
  return(MSEs)
}

Plot.SplineCV <- function(MSEs, dfs) {
  Rounds <- ncol(MSEs)
  mean_mse <- rowMeans(MSEs)
  se_mse   <- apply(MSEs, 1, sd) / sqrt(Rounds)
  
  lower <- mean_mse - 2 * se_mse
  upper <- mean_mse + 2 * se_mse
  
  mat <- cbind(mean_mse, lower, upper)
  
  matplot(dfs, mat, type = "b",
          pch = c(19, NA, NA), lty = c(1, 2, 2),
          col = c("black", "gray", "gray"),
          xlab = "Spline df",
          ylab = "CV MSE",
          main = paste0("10-fold CV: MSE vs Spline df"))
  legend("topright",
         legend = c("Mean MSE", "±2 × SE"),
         lty = c(1, 2), col = c("black", "gray"), bty = "n")
}

# 1. Define predictors & response
predictors <- setdiff(names(Data.train), "Y")
response   <- "Y"

# 2. Run CV over dfs = 2 through 6
dfs       <- 2:6
cv_mses   <- CV.spline.f(
  data     = Data.train,
  preds    = predictors,
  response = response,
  dfs      = dfs,
  k        = 10,
  Rounds   = 3
)

# 3. Plot to choose best df
Plot.SplineCV(cv_mses, dfs)

# 4. Best df:
mean_mses <- rowMeans(cv_mses)
best_df   <- dfs[which.min(mean_mses)]
cat("Best spline df (by CV):", best_df, "\n")
print(mean_mses)
