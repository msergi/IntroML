
#######################################################################
# Setup & loading
#######################################################################

setwd('~/Desktop/intro-ml/houseprices')
library(data.table)

train <- fread('train.csv', stringsAsFactor = T)
test <- fread('test.csv', stringsAsFactors = T)
# Target: SalePrice

dim(train)
dim(test)

head(train)
head(test)

# Number of NAs in train and test
nas <- apply(train, 2, function(x) {sum(is.na(x))})
nas[nas > 0]
nas <- apply(test, 2, function(x) {sum(is.na(x))})
nas[nas > 0]

# Although represented with numbers, some columns are categorical
catcols <- c('MSSubClass', 'OverallQual', 'OverallCond')
train[, (catcols) := lapply(.SD, as.factor), .SDcols = catcols]
test[, (catcols) := lapply(.SD, as.factor), .SDcols = catcols]


#######################################################################
# Data Cleaning
#######################################################################

# Which variables are numeric?
filt <- unlist(lapply(train, is.numeric))
filt <- filt & !(names(train) %in% c('Id', 'SalePrice'))
numeric_vars <- names(train)[filt]
numeric_vars


# Plotting distributions of numeric variables
pdf(file = '01-densityplots.pdf')
for(nvar in c(numeric_vars, 'SalePrice')){
  temp <- unlist(train[, ..nvar])
  temp <- temp[!(is.na(temp))]
  
  plot(density(temp),
       main = nvar,
       col = 'cornflowerblue')
}
dev.off()

# Scatter plots against SalePrice
pdf(file = '02-scatterplots-saleprice.pdf')
for(nvar in numeric_vars) {
  plot(unlist(train[, ..nvar]),
       unlist(train[, .(SalePrice)]),
       pch = 20,
       col = 'cornflowerblue',
       xlab = nvar,
       ylab = 'SalePrice')
}
dev.off()


# Which variables are categorical?
filt <- unlist(lapply(fulldata, is.factor))
cats <- names(fulldata)[filt]
cats

# Boxplots against SalePrice
pdf(file = '03-boxplots-saleprice.pdf')
for(cat in cats) {
  x <- unlist(train[, ..cat])
  y <- unlist(train[, .(SalePrice)])
  boxplot(y ~ x,
          xlab = cat,
          ylab = 'SalePrice',
          las = 2)
}
dev.off()


# I'm going to work with full dataset from now on
test[, 'SalePrice' := NA] # Now can be binded (same n of cols)
fulldata <- rbindlist(list(train, test))
dim(fulldata)

# Which numeric variables have missing values?
numsna <- apply(fulldata[,..numeric_vars], 2, function(x) {sum(is.na(x))})
numsna[numsna > 0]

# Which categories do have missing values?
catsna <- apply(fulldata[,..cats], 2, function(x) {sum(is.na(x))})
catsna[catsna > 0]


### FILLING MISSING DATA

# Filling some columns 'None' based on data description
colsnone <- c('Alley', 'MasVnrType', 'FireplaceQu', 'GarageType',
              'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC',
              'MiscFeature', 'Fence', 'MasVnrType','BsmtQual',
              'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
              'BsmtFinType2')
myfillna <- function(DT, columns = names(DT), fill = NA) {
  # Fills missing values from a DT with a specified value and
  # indicating columns (default all columns and NA filling)
  for (i in columns)
    DT[is.na(get(i)), (i):=fill]
}

myfillna(fulldata, colsnone, 'None')

fulldata[, Utilities := NULL] # Useless (unique categories in train)

# Filling with most common value
fulldata[is.na(Functional), Functional := 'Typ']
fulldata[is.na(Electrical), Electrical := 'SBrkr']
fulldata[is.na(Exterior1st), Exterior1st := 'VinylSd']
fulldata[is.na(Exterior2nd), Exterior2nd := 'VinylSd']
fulldata[is.na(KitchenQual), KitchenQual := 'TA']
fulldata[is.na(SaleType), SaleType := 'WD']
fulldata[is.na(MSZoning), MSZoning := 'RL']

colszero <- c('GarageYrBlt', 'GarageCars', 'GarageArea',
              'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
              'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
              'MasVnrArea')

myfillna(fulldata, colszero, 0)

# LotFrontage: median by group
fulldata[, temp := median(LotFrontage, na.rm = T), by = 'Neighborhood']
fulldata[is.na(LotFrontage), LotFrontage := temp]
fulldata[, temp := NULL]

sum(is.na(fulldata)) == dim(test)[1] # TRUE


# Checking correlations
nums <- unlist(lapply(fulldata, is.numeric))
col <- colorRampPalette(c('yellow', 'black', 'cyan'))(256)
corr <- round(cor(fulldata[Id %in% trainId, ..nums]), 2)
corr2 <- (round(cor(fulldata[Id %in% trainId, ..nums],
                    method = 'spearman'), 2))

pdf('04-pearson-corr.pdf')
heatmap(corr, col = col, symm = T)
dev.off()

pdf('05-spearman-corr.pdf')
heatmap(corr2, col = col, symm = T)
dev.off()

train.data <- fulldata[Id %in% trainId]
test.data <- fulldata[Id %in% testId]

# Outliers removal based on scatter plots
train.data <- train.data[LotFrontage < 300]
train.data <- train.data[LotArea < 100000]
train.data <- train.data[BsmtFinSF1 < 5000]
train.data <- train.data[BsmtFinSF2 < 1400]
train.data <- train.data[TotalBsmtSF < 5000]
train.data <- train.data['1stFlrSF' < 4000]
train.data <- train.data[!(GrLivArea > 4000 & SalePrice < 300000)]
train.data <- train.data[MiscVal < 5000]
dim(train.data)


#######################################################################
# Modeling
#######################################################################

library(h2o)
h2o.init(nthreads = 1)
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
# MSE:  694751153
# RMSE:  26358.13
# MAE:  16868.22
# RMSLE:  0.1521004
# Mean Residual Deviance :  694751153
# R^2 :  0.8906238

results <- as.data.frame(h2o.cbind(test$SalePrice, p$predict))
head(results)

gbm_params1 <- list(learn_rate = c(0.05, 0.1),
                    max_depth = c(3, 4, 5),
                    sample_rate = c(0.8, 1.0),
                    col_sample_rate = c(0.1, 0.2))


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

#################
# xgboost

xgb_params <- list(learn_rate = c(0.1, 0.3, 0.5),
                   max_depth = c(3, 6, 9),
                   sample_rate = c(0.5, 0.8, 1),
                   col_sample_rate = c(0.5, 0.8, 1))

xgb_grid <- h2o.grid("xgboost", x = x, y = y,
                      grid_id = "xgb",
                      training_frame = train,
                      validation_frame = valid,
                      ntrees = 100,
                      seed = 2020,
                      hyper_params = xgb_params)

xgb_perf <- h2o.getGrid(grid_id = "xgb",
                         sort_by = "rmse",
                         decreasing = FALSE)
xgb_perf
best_xgb <- h2o.getModel(xgb_perf@model_ids[[1]])
p <- h2o.predict(best_xgb, test)

h2o.performance(model = best_xgb, newdata = test)
results <- h2o.cbind(as.h2o(results), p$predict)

## Result export
tk <- as.h2o(test.kaggle)

p2 <- h2o.predict(best_xgb, tk[2:80])
r <- as.data.frame(h2o.cbind(tk$Id, p2$predict))
colnames(r) <- c('Id', 'SalePrice')

write.csv(r, file = 'sub.csv', quote = F, row.names = F)
