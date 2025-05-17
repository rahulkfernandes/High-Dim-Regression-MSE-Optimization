load("./data/Data.train.update.RData")
load("./data/Data.test.update.RData")

# Degree of polynomial
deg <- 3 

# Predictor names (exclude Y)
predictors <- setdiff(names(Data.train), "Y")

# Build polynomial terms
poly_terms <- sapply(predictors, function(p) paste0("poly(", p, ", ", deg, ", raw = FALSE)"))

# Full formula                           
fmla <- as.formula(paste("Y ~", paste(poly_terms, collapse = " + ")))

# Fit model                            
poly_mod <- lm(fmla, data = Data.train)
summary(poly_mod)
par(mfrow = c(2, 2)); plot(poly_mod)

predictions <- predict(poly_mod, newdata = Data.test)

# Details
last_name <- "Fernandes"
mse_guess <- 10

# Create a character vector for output
output_lines <- c(
  last_name,
  as.numeric(mse_guess),
  as.numeric(predictions)
)

# Write to CSV
writeLines(output_lines, "Fernandes.Predictions.csv")
