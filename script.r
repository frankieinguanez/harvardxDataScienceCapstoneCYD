
# Dry Beans Dataset from: https://archive.ics.uci.edu/ml/datasets/Dry+Bean+dataset

##########################################################
# Global Variables
##########################################################
options(digits = 5)

##########################################################
# Data acquisition and setup
##########################################################

# clear everything
rm(list=ls())

# Download libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(psych)) install.packages("psych", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(psych)
library(caret)
library(randomForest)
library(xgboost)

dl <- tempfile()
download.file("https://github.com/frankieinguanez/harvardxDataScienceCapstoneCYD/raw/main/dryBean.csv", dl)

d <- read.csv2(dl,header = TRUE, sep = ",", stringsAsFactors = FALSE)

# Test set will be 20% 
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test.index <- createDataPartition(y = d$Class, times = 1, p = 0.2, list = FALSE)
beans <- d[-test.index,]
test <- d[test.index,]

# Clean-up
rm(list=c("d", "test.index", "dl"))

# Save data image
save.image("dryBeans.RData")

##########################################################
# Exploratory Analysis
##########################################################

dim(beans)
str(beans)
summary(beans)

##########################################################
# Data clean & transformation
##########################################################

beans$Perimeter <- as.numeric(beans$Perimeter)
beans$MajorAxisLength <- as.numeric(beans$MajorAxisLength)
beans$MinorAxisLength <- as.numeric(beans$MinorAxisLength)
beans$AspectRation <- as.numeric(beans$AspectRation)
beans$Eccentricity <- as.numeric(beans$Eccentricity)
beans$ConvexArea <- as.numeric(beans$ConvexArea)
beans$EquivDiameter <- as.numeric(beans$EquivDiameter)
beans$Extent <- as.numeric(beans$Extent)
beans$Solidity <- as.numeric(beans$Solidity)
beans$roundness <- as.numeric(beans$roundness)
beans$Compactness <- as.numeric(beans$Compactness)
beans$ShapeFactor1 <- as.numeric(beans$ShapeFactor1)
beans$ShapeFactor2 <- as.numeric(beans$ShapeFactor2)
beans$ShapeFactor3 <- as.numeric(beans$ShapeFactor3)
beans$ShapeFactor4 <- as.numeric(beans$ShapeFactor4)

beans$Class <- as.factor(beans$Class)

summary(beans)

# Min Max scaling
process <- preProcess(as.data.frame(beans), method=c("range"))
beans <- predict(process, as.data.frame(beans))

summary(beans)

# Clean-up
rm(process)

# Save data image
save.image("dryBeans.RData")

##########################################################
# Data visualisation
##########################################################

beans %>%
  ggplot(aes(Class)) +
  geom_histogram(color=I("black"), stat="count") +
  scale_y_continuous() +
  labs(title = "Bean Class distribution", x = "Class", y = "Count")


boxplot(Area~Class, data=beans)
boxplot(Perimeter~Class, data=beans)
boxplot(MajorAxisLength~Class, data=beans)
boxplot(MinorAxisLength~Class, data=beans)
oxplot(ConvexArea~Class, data=beans)
boxplot(EquivDiameter~Class, data=beans)

boxplot(AspectRation~Class, data=beans)

boxplot(Eccentricity~Class, data=beans)
boxplot(Solidity~Class, data=beans)
boxplot(roundness~Class, data=beans)

boxplot(Extent~Class, data=beans)

boxplot(Compactness~Class, data=beans)

boxplot(ShapeFactor1~Class, data=beans)
boxplot(ShapeFactor2~Class, data=beans)
boxplot(ShapeFactor3~Class, data=beans)
boxplot(ShapeFactor4~Class, data=beans)

##########################################################
# Data Modelling - This section takes very long to execute
##########################################################

# Data splitting

# Val set will be 20%
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
val.index <- createDataPartition(y = beans$Class, times = 1, p = 0.2, list = FALSE)
d.train <- beans[-val.index,]
d.val <- beans[val.index,]

# Clean-up
rm(list=c("val.index"))

# Defining cross validation
repeat_cv <- trainControl(method='repeatedcv', 
                          number=5, 
                          repeats=3, 
                          allowParallel = TRUE,
                          verboseIter = FALSE,
                          classProbs = TRUE)

# KNN using
knn <- train(Class ~ .,  
             method = "knn", 
             tuneGrid = data.frame(k = seq(1, 50, 2)), 
             trControl = repeat_cv,
             metric='Accuracy', 
             data = d.train)
plot(knn)
knn

# Random Forest
rf <- train(Class ~ .,
            method = "rf",
            tuneLength  = 15,
            trControl = repeat_cv,
            metric='Accuracy', 
            data = d.train)
plot(rf)
rf

# XGBoost
xgbGrid <- expand.grid(nrounds = c(100,200),
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree = seq(0.5, 0.9, length.out = 5),
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1)

xgb <- train(Class ~ .,
             method = "xgbTree",
             tuneGrid  = xgbGrid,
             trControl = repeat_cv,
             metric='Accuracy', 
             data = d.train)
plot(xgb)
xgb

# Model evaluation to determine ideal using validation dataset
confusionMatrix(predict(knn, d.val), d.val$Class)
confusionMatrix(predict(rf, d.val), d.val$Class)
confusionMatrix(predict(xgb, d.val), d.val$Class)

# Clean-up
rm(list=c("xgbGrid", "repeat_cv"))

# Save data image
save.image("dryBeans.RData")

##########################################################
# Final evaluation
##########################################################

# Clean testing dataset
test$Perimeter <- as.numeric(test$Perimeter)
test$MajorAxisLength <- as.numeric(test$MajorAxisLength)
test$MinorAxisLength <- as.numeric(test$MinorAxisLength)
test$AspectRation <- as.numeric(test$AspectRation)
test$Eccentricity <- as.numeric(test$Eccentricity)
test$ConvexArea <- as.numeric(test$ConvexArea)
test$EquivDiameter <- as.numeric(test$EquivDiameter)
test$Extent <- as.numeric(test$Extent)
test$Solidity <- as.numeric(test$Solidity)
test$roundness <- as.numeric(test$roundness)
test$Compactness <- as.numeric(test$Compactness)
test$ShapeFactor1 <- as.numeric(test$ShapeFactor1)
test$ShapeFactor2 <- as.numeric(test$ShapeFactor2)
test$ShapeFactor3 <- as.numeric(test$ShapeFactor3)
test$ShapeFactor4 <- as.numeric(test$ShapeFactor4)

test$Class <- as.factor(test$Class)

summary(test)

# Min Max scaling
process <- preProcess(as.data.frame(test), method=c("range"))
test <- predict(process, as.data.frame(test))

summary(test)

# Clean-up
rm(process)

# Apply best model to test dataset
confusionMatrix(predict(xgb, test), test$Class)

save.image("dryBeans.RData")
