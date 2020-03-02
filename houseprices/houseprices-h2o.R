
#######################################################################
# Loading libraries and data
#######################################################################

library(data.table)
library(mice)

setwd('~/Desktop/intro-ml/houseprices')

# hp <- read.csv('train.csv') # house prices
hp <- fread('train.csv')

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
nums <- unlist(lapply(hp, is.numeric))
names(hp)[nums]

# Although represented with numbers, some columns are categorical
# Conversion:
catcols <- c('MSSubClass', 'OverallQual', 'OverallCond')
hp[, (catcols) := lapply(.SD, as.factor), .SDcols = catcols]

# Which numeric variables do have missing values?
numsna <- apply(hp[,..nums], 2, function(x) {sum(is.na(x))})
numsna <- names(hp[,..nums])[numsna > 0]
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
par(mfrow = c(1, 2))
heatmap(corr, col = col, symm = T)
heatmap(corr2, col = col, symm = T)

install.packages('car')
scatterplotMatrix

## CATEGORICAL VARIABLES

# Which variables are categorical?
cats <- unlist(lapply(hp, is.factor))
names(hp)[cats]

# Which categories do have missing values?
catsna <- apply(hp[,cats], 2, function(x) {sum(is.na(x))})
catsna <- names(hp[,cats])[catsna > 0]
catsna
summary(hp[catsna])

hp[is.na(PoolQC), PoolQC := 'NoPool']
hp[is.na(Fence), Fence := 'NoFence']

#######################################################################
# Training
#######################################################################

library(h2o)
h2o.init()

parts <- h2o.split(hp, c(0.6, 0.2) )
train <- parts[[1]]
valid <- parts[[2]]
test <- parts[[3]]
rm(parts)