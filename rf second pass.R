#Load data obtained from stated source
traindata <-read.csv("pml-training.csv")
testdata <-read.csv("pml-testing.csv")

#Load required Libraries
library(lattice)
library(ggplot2)
library(caret)
library(randomForest)
library(pracma)

#poll size of data set to ensure data loaded correctly
dim(traindata)
names(traindata)

#Set Seed for repeatable results
set.seed(555)

#Compact data by dropping time data and label columns EXCEPT for classe, additionally there are 
#numerous blank columns that can be removed as neither impact the model
clean1traindata<-traindata[,-c(1:7,12:36,50:59,69:83,87:101,103:112,125:139,141:150)]

#test data for near zero variance NZV in variables, NZV variables reported or all required
nzvtest <- nearZeroVar(clean1traindata, saveMetrics = TRUE)
if (any(nzvtest$nzvtest)) nzvtest else message("All variables required")

#split data into two sets - 60% for training and 40% for verification
mytrain <- createDataPartition(y=clean1traindata$classe, p=0.6, list =FALSE)
trainset <- clean1traindata[mytrain,]
verifyset <- clean1traindata[-mytrain,]
names(clean1traindata)
dim(trainset)
dim(verifyset)

#build model using Random Forest and measure function time using tic toc
tic()
rfmodel <- train(trainset$classe ~ .,data = trainset, method = "rf")
toc()

#show model then run a Confusion Matrix on the verification set which is a subset 40% of the main data
rfmodel
predictset <- predict(rfmodel, verifyset)
confusionMatrix(predictset,verifyset$classe)

#run Variable Importance
varImp(rfmodel)

#run OOB error estimate
rfmodel$finalModel

#run model against testdata the pml test set to determine the exercise class
finaltest <- predict(rfmodel, testdata)
finaltest
