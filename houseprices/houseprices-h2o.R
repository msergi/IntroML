
#######################################################################
# Loading libraries and data
#######################################################################

library(data.table)
library(mice)

setwd('~/Desktop/intro-ml/houseprices')

# hp <- read.csv('train.csv') # house prices
hp <- fread('train.csv', stringsAsFactor = T)

dim(hp)
head(hp)
summary(hp)
nas <- apply(hp, 2, function(x) {sum(is.na(x))})
nas[nas > 0]
# Target: Sales Price

hp <- hp[,-1] # Removing Id column
dim(hp)
# 79 features + 1 target

#######################################################################
# Data Cleaning
#######################################################################

## NUMERICAL VARIABLES

# Which variables are numeric?
filt <- unlist(lapply(hp, is.numeric))
nums <- names(hp)[filt]

# Although represented with numbers, some columns are categorical
# Conversion:
catcols <- c('MSSubClass', 'OverallQual', 'OverallCond')
hp[, (catcols) := lapply(.SD, as.factor), .SDcols = catcols]

# Which numeric variables do have missing values?
numsna <- apply(hp[,..nums], 2, function(x) {sum(is.na(x))})
numsna <- names(numsna[numsna > 0])
numsna
# 'LotFrontage' 'MasVnrArea'  'GarageYrBlt'

# Imputation 
plot(density(hp$LotFrontage, na.rm = T), main = 'LotFrontage density')

imputed <- mice(hp[,..numsna],
                m = 5, 
                maxit = 50, 
                method = 'pmm',
                seed = 2020)

complete.data <- mice::complete(imputed)
for (coln in numsna){
    hp[, (coln) := complete.data[, coln]]
}

# Checking correlations
nums <- unlist(lapply(hp, is.numeric))
col <- colorRampPalette(c('yellow', 'black', 'cyan'))(256)
corr <- round(cor(hp[,..nums]), 2)
corr2 <- (round(cor(hp[,..nums], method = 'spearman'), 2))

pdf('01-pearson-corr.pdf')
heatmap(corr, col = col, symm = T)
dev.off()

pdf('02-spearman-corr.pdf')
heatmap(corr2, col = col, symm = T)
dev.off()

## CATEGORICAL VARIABLES

# Which variables are categorical?
filt <- unlist(lapply(hp, is.factor))
cats <- names(hp)[filt]

# Which categories do have missing values?
catsna <- apply(hp[,..cats], 2, function(x) {sum(is.na(x))})
catsna <- names(catsna[catsna > 0])
catsna
summary(hp[,..catsna])

fullcats <- cats[!(cats %in% catsna)] # No NAs subset
calc_anova <- function(x) {
    an <- anova(lm(hp[, SalePrice] ~ x))
    an[1,5]
}

res <- hp[, lapply(.SD, calc_anova), .SDcols = fullcats]
data.frame("p-value" = unlist(res),
           "significant?" = (unlist(res) < 0.05))

sprice <- hp[, SalePrice]
par(mfrow = c(1, 3))
boxplot(sprice ~ hp[, Street]) # Similar dist
boxplot(sprice ~ hp[, Utilities]) # Low counts in 1 group (1)
boxplot(sprice ~ hp[, LandSlope]) # Similar dist

# Handling missing values
myfillna = function(DT, columns = names(DT), fill = NA) {
  for (i in columns)
    DT[is.na(get(i)), (i):=fill]
}

myfillna(hp, catsna, 'None')

any(is.na(hp))

#######################################################################
# Testing
#######################################################################

## NUMERIC VARIABLES

test <- fread('test.csv', stringsAsFactor = T)

filt <- unlist(lapply(test, is.numeric))
nums <- names(test)[filt]

catcols <- c('MSSubClass', 'OverallQual', 'OverallCond')
test[, (catcols) := lapply(.SD, as.factor), .SDcols = catcols]

# Which numeric variables do have missing values?
numsna <- apply(test[,..nums], 2, function(x) {sum(is.na(x))})
numsna <- names(numsna[numsna > 0])
numsna
# 'LotFrontage' 'MasVnrArea'  'GarageYrBlt'

imputed <- mice(test[,..numsna], m = 5, maxit = 50, method = 'pmm')

complete.data <- mice::complete(imputed)
for (coln in numsna){
    test[, (coln) := complete.data[, coln]]
}

## CATEGORICAL VARIABLES

# Which variables are categorical?
filt <- unlist(lapply(test, is.factor))
cats <- names(test)[filt]

# Which categories do have missing values?
catsna <- apply(test[,..cats], 2, function(x) {sum(is.na(x))})
catsna <- names(catsna[catsna > 0])
catsna
summary(test[,..catsna])

# Handling missing values
myfillna = function(DT, columns = names(DT), fill = NA) {
  for (i in columns)
    DT[is.na(get(i)), (i):=fill]
}

myfillna(test, catsna, 'None')
any(is.na(test))

#######################################################################
# Training
#######################################################################

library(h2o)
h2o.init(nthreads = 2)
hp <- as.h2o(hp)

parts <- h2o.splitFrame(hp, c(0.6, 0.2), seed = 2020)
train <- parts[[1]]
valid <- parts[[2]]
test <- parts[[3]]
rm(parts)

y <- 'SalePrice'
x <- setdiff(names(hp), y)

m <- h2o.gbm(x, y, training_frame = train, validation_frame = valid)
p <- h2o.predict(m, test)

h2o.performance(model = m, newdata = test)

results <- as.data.frame(h2o.cbind(test$SalePrice, p$predict))
head(results)

gbm_params1 <- list(learn_rate = c(0.01, 0.1),
                    max_depth = c(4, 5, 8),
                    sample_rate = c(0.8, 1.0),
                    col_sample_rate = c(0.1, 0.2, 0.5))


gbm_grid1 <- h2o.grid("gbm", x = x, y = y,
                      grid_id = "gbm_grid2",
                      training_frame = train,
                      validation_frame = valid,
                      ntrees = 100,
                      seed = 1,
                      hyper_params = gbm_params1,
                      nfolds = 5)

# Get the grid results, sorted by validation AUC
gbm_gridperf1 <- h2o.getGrid(grid_id = "gbm_grid2",
                             sort_by = "rmse",
                             decreasing = FALSE)
print(gbm_gridperf1)

# Grab the top GBM model, chosen by validation AUC
best_gbm1 <- h2o.getModel(gbm_gridperf1@model_ids[[1]])

# Now let's evaluate the model performance on a test set
# so we get an honest estimate of top model performance
best_gbm_perf1 <- h2o.performance(model = best_gbm1,
                                  newdata = test)
h2o.mse(best_gbm_perf1)

## Prediction



# Look at the hyperparameters for the best model
print(best_gbm1@model[["model_summary"]])