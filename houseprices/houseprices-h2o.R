
#######################################################################
# Loading libraries and data
#######################################################################

library(data.table)
library(h2o)
library(mice)

setwd('~Desktop/ml/houseprices')

hp <- read.csv('train.csv')
dim(hp)
head(hp)

summary(hp)
#apply(hp, 2, function(x) {sum(is.na(x))})

# Target: Sales Price

hp <- hp[,-1] # Removing Id column

#######################################################################
# Data Cleaning
#######################################################################

## NUMERICAL VARIABLES

# Which variables are numeric?
nums <- unlist(lapply(hp, is.numeric))
names(hp)[nums]

# Although represented with numbers, some columns are categorical
hp$MSSubClass <- as.factor(hp$MSSubClass)
hp$OverallQual <- as.factor(hp$OverallQual)
hp$OverallCond <- as.factor(hp$OverallCond)

# Which numeric variables do have missing values?
numsna <- apply(hp[,nums], 2, function(x) {sum(is.na(x))})
numsna <- names(hp[,nums])[numsna > 0]
numsna
# "LotFrontage" "MasVnrArea"  "GarageYrBlt"

## Imputation 
plot(density(hp$LotFrontage, na.rm = T))


imputed <- mice(hp[,numsna],
                m = 5, 
                maxit = 50, 
                method = "pmm",
                seed = 2020)

complete.data <- mice::complete(imputed)
hp$LotFrontage <- complete.data$LotFrontage
hp$MasVnrArea <- complete.data$MasVnrArea
hp$GarageYrBlt <- complete.data$GarageYrBlt

# Checking correlations
corr <- round(cor(hp[,nums]), 2)
col <- colorRampPalette(c("yellow", "black", "cyan"))(256)
heatmap(corr, col=col, symm=TRUE)

install.packages('car')
scatterplotMatrix

## CATEGORICAL VARIABLES

# Which variables are categorical?
cats <- unlist(lapply(hp, is.factor))
names(hp)[cats]

# Which categories do have missing values?
catsna <- apply(hp[,cats], 2, function(x) {sum(is.na(x))})
catsna <- names(hp[,cats])[numsna > 0]
catsna

########################################
# Training
########################################

h2o.init()

parts <- h2o.split(hp, c(0.6, 0.2) )
train <- parts[[1]]
valid <- parts[[2]]
test <- parts[[3]]
rm(parts) #Optional
