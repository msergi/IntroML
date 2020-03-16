
#######################################################################
# Setup & loading
#######################################################################

setwd('C:/Users/Sergi/Desktop/intro-ml/houseprices')
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
dir.create('plots')

pdf(file = './plots/01-densityplots.pdf')
for(nvar in c(numeric_vars, 'SalePrice')){
  temp <- unlist(train[, ..nvar])
  temp <- temp[!(is.na(temp))]
  
  plot(density(temp),
       main = nvar,
       col = 'cornflowerblue')
}
dev.off()

# Scatter plots against SalePrice
pdf(file = './plots/02-scatterplots-saleprice.pdf')
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
filt <- unlist(lapply(train, is.factor))
cats <- names(train)[filt]
cats

# Boxplots against SalePrice
pdf(file = './plots/03-boxplots-saleprice.pdf')
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

trainId <- train[, Id]
testId <- test[, Id]

# Which numeric variables have missing values?
numsna <- apply(fulldata[,..numeric_vars], 2, function(x) {sum(is.na(x))})
numsna[numsna > 0]

# Which categories have missing values?
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

fulldata[, Utilities := NULL] # Useless column (exclusive categories in train)

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
median_neigh <- fulldata[, median(LotFrontage, na.rm=T), by = Neighborhood]
for (neigh in median_neigh$Neighborhood) {
  m <- median_neigh[Neighborhood == neigh, V1]
  fulldata[is.na(LotFrontage) & Neighborhood == neigh, LotFrontage := m]
}

sum(is.na(fulldata)) == dim(test)[1] # TRUE


# Checking correlations
nums <- unlist(lapply(fulldata, is.numeric))
col <- colorRampPalette(c('yellow', 'black', 'cyan'))(256)
corr <- round(cor(fulldata[Id %in% trainId, ..nums]), 2)
corr2 <- (round(cor(fulldata[Id %in% trainId, ..nums],
                    method = 'spearman'), 2))

pdf('./plots/04-pearson-corr.pdf')
heatmap(corr, col = col, symm = T)
dev.off()

pdf('./plots/05-spearman-corr.pdf')
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
h2o.init()
h2o.removeAll()
train.data$Id <- NULL
train.data <- as.h2o(train.data)

# parts <- h2o.splitFrame(train.data, c(0.6, 0.2), seed = 2020)
# train <- parts[[1]]
# valid <- parts[[2]]
# test <- parts[[3]]
# rm(parts)

parts <- h2o.splitFrame(train.data, c(0.2), seed = 2020)
train <- parts[[2]]
valid <- parts[[1]]
rm(parts)

y <- 'SalePrice'
x <- setdiff(names(train), y)


### GLM

glm_params <- list(alpha = c(0, 0.25, 0.5, 0.75, 1),
                   lambda = c(0.001, 0.01, 0.1, 1, 10, 100))

glm_grid <- h2o.grid('glm', x = x, y = y,
                     grid_id = 'glm1',
                     training_frame = train,
                     validation_frame = valid,
                     hyper_params = glm_params,
                     nfolds = 10,
                     fold_assignment = 'Modulo',
                     keep_cross_validation_predictions = T)

glm_perf <- h2o.getGrid(grid_id = "glm1",
                        sort_by = "rmsle",
                        decreasing = FALSE)
glm_perf
best_glm <- h2o.getModel(glm_perf@model_ids[[1]])
# h2o.performance(best_glm, newdata = test)


### Random forest

rf <- h2o.randomForest(x = x, y = y,
                       training_frame = train,
                       validation_frame = valid,
                       ntrees = 100,
                       max_depth = 30,
                       sample_rate = 1,
                       seed = 2020,
                       nfolds = 10,
                       fold_assignment = 'Modulo',
                       keep_cross_validation_predictions = T)
# h2o.performance(model = rf, newdata = test)


### GBM

gbm_params <- list(learn_rate = c(0.05, 0.1),
                    sample_rate = c(0.8, 1.0),
                    col_sample_rate = c(0.1, 0.2))


gbm_grid <- h2o.grid("gbm", x = x, y = y,
                     grid_id = "gbm1",
                     training_frame = train,
                     validation_frame = valid,
                     ntrees = 100,
                     seed = 1,
                     hyper_params = gbm_params,
                     nfolds = 10,
                     fold_assignment = 'Modulo',
                     keep_cross_validation_predictions = T)

gbm_perf <- h2o.getGrid(grid_id = 'gbm1',
                        sort_by = 'rmsle',
                        decreasing = FALSE)
print(gbm_perf)
best_gbm <- h2o.getModel(gbm_perf@model_ids[[1]])
# h2o.performance(best_gbm, newdata = test)


### Neural networks (default settings)

DL <- h2o.deeplearning(x = x, y = y, model_id = 'DL',
                       training_frame = train,
                       validation_frame = valid,
                       nfolds = 10,
                       fold_assignment = 'Modulo',
                       keep_cross_validation_predictions = T)

### Ensembling

models <- list(best_glm, rf, best_gbm, DL)

ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                training_frame = train,
                                model_id = 'ensemble',
                                base_models = models)

# h2o.performance(ensemble, newdata = test)


#######################################################################
# Predict & export
#######################################################################

test.data <- as.h2o(test.data)

p <- h2o.predict(ensemble, test.data[2:79])
r <- as.data.frame(h2o.cbind(test.data$Id, p$predict))
colnames(r) <- c('Id', 'SalePrice')

write.csv(r, file = 'sub.csv', quote = F, row.names = F)
