library(caret);library(ggplot2);library(rattle)

# set working directory
setwd("~/work/courses/Practical Machine Learning/project")
# read training datatset
trainread <- read.csv("pml-training.csv", na.strings=c("NA", "#DIV/0!"))

# read testing dataset
testread <- read.csv("pml-testing.csv", na.strings=c("NA", "#DIV/0!"))

# first 7 variables are bookkeeping data, restrict to 8:160
testtmp <- testread[,8:160]

# find a subset of variables that are complete in the testing dataset
allmissing = lapply(lapply(testtmp[, ], is.na), sum)
table(as.numeric(allmissing))
selGood <- (as.numeric(allmissing))==0
testtotal <- testtmp[,selGood]

# use the same subset of variables in the training dataset
traintotaltmp <- trainread[,8:160]
traintotal <- traintotaltmp[,selGood]

# compare different algorithms
acc <- data.frame(names=c("CART", "bagged CART", "random forrest", "boost"), avgtestacc=numeric(4), sdtestacc=numeric(4), avgvalidacc=numeric(4))

# select a subsample of all data
set.seed(543534)
nameid <- "testrun"
compute <- FALSE
numdata <- 2000
numvalid <- 2000
print(nrow(traintotal))
outTotalSelection <- createDataPartition(traintotal$classe, p = 1 - (numdata+numvalid)/nrow(traintotal), list=FALSE)
#totalselection <- randomSample(traintotal, numdata + numvalid)
totalselection <- traintotal[-outTotalSelection,]

inSelection <- createDataPartition(totalselection$classe, p=numdata / (numdata + numvalid), list=FALSE)
selection <- totalselection[inSelection,]

# save another sample for final validation
validation <- totalselection[-inSelection,]
print(paste(nrow(selection),nrow(validation)))

# use K-folds
numfolds <- 10
foldSel <- createFolds(y = selection$classe, k = numfolds, returnTrain = TRUE)

testacclist <- data.frame(cart=numeric(numfolds), rf=numeric(numfolds), bcart=numeric(numfolds), gbm=numeric(numfolds))
validacclist <- data.frame(cart=numeric(numfolds), rf=numeric(numfolds), bcart=numeric(numfolds), gbm=numeric(numfolds))

# perform training and testing on all folds
for (i in 1:numfolds) {
    
# divide dataset
    training <- selection[foldSel[[i]],]
    testing <- selection[-foldSel[[i]],]
    print(paste("Training: ",nrow(training)," Testing: ",nrow(testing)))
# CART
    set.seed(543534)
    outname=paste(nameid,"cart",i,"Rdata",sep=".")
    if (!compute) {
        load(outname)
    } else {
        tmpfit <- train(classe ~ ., method="rpart", data = training)
        save(tmpfit, file=outname)
    }
    testPred <- predict(tmpfit, newdata = testing)    
    testacclist$cart[i] <- mean(testPred == testing$classe)
    validPred <- predict(tmpfit, newdata=validation)
    validacclist$cart[i] <- mean(validPred == validation$classe)

# bCART
    set.seed(543534)
    outname=paste(nameid,"bcart",i,"Rdata",sep=".")
    if (!compute) {
        load(outname)
    } else {
        tmpfit <- train(classe ~ ., method="treebag", data = training)
        save(tmpfit, file=outname)
    }
    testPred <- predict(tmpfit, newdata = testing)    
    testacclist$bcart[i] <- mean(testPred == testing$classe)
    validPred <- predict(tmpfit, newdata=validation)
    validacclist$bcart[i] <- mean(validPred == validation$classe)

 # rf
    set.seed(543534)
    outname=paste(nameid,"rf",i,"Rdata",sep=".")
    if (!compute) {
        load(outname)
    } else {
        tmpfit <- train(classe ~ ., method="rf", data = training)
        save(tmpfit, file=outname)
    }
    testPred <- predict(tmpfit, newdata = testing)    
    testacclist$rf[i] <- mean(testPred == testing$classe)
    validPred <- predict(tmpfit, newdata=validation)
    validacclist$rf[i] <- mean(validPred == validation$classe)

# gbm
    set.seed(543534)
    outname=paste(nameid,"gbm",i,"Rdata",sep=".")
    if (!compute) {
        load(outname)
    } else {
        tmpfit <- train(classe ~ ., method="gbm", data = training, verbose=FALSE)
        save(tmpfit, file=outname)
    }
    testPred <- predict(tmpfit, newdata = testing)    
    testacclist$gbm[i] <- mean(testPred == testing$classe)
    validPred <- predict(tmpfit, newdata=validation)
    validacclist$gbm[i] <- mean(validPred == validation$classe)

# inform
    print(paste(i, testacclist$cart[i], testacclist$bcart[i], testacclist$rf[i], testacclist$gbm[i]))
    print(paste(i, validacclist$cart[i], validacclist$bcart[i], validacclist$rf[i], validacclist$gbm[i]))

}

# compute average and standard deviation of testing and validation accuracies

acc$avgtestacc <- c(mean(testacclist$cart), mean(testacclist$bcart), mean(testacclist$rf), mean(testacclist$gbm))
acc$sdtestacc <- c(sd(testacclist$cart), sd(testacclist$bcart), sd(testacclist$rf), sd(testacclist$gbm))


# perform validation using the best fit of each method

bestcart <- which.max(testacclist$cart)
bestbcart <- which.max(testacclist$bcart)
bestrf <- which.max(testacclist$rf)
bestgbm <- which.max(testacclist$gbm)
print(paste(bestcart, bestbcart, bestrf, bestgbm))
bests<-c(cart=bestcart, bcart=bestbcart, rf=bestrf, gbm=bestgbm)
acc$avgvalidacc <- c(validacclist$cart[bestcart], validacclist$bcart[bestbcart], validacclist$rf[bestrf], validacclist$gbm[bestgbm])

print(acc)

dput(acc, paste(nameid,"acc.R",sep="."))
dput(bests, paste(nameid,"bests.R",sep="."))

# predict testing data

# best model: rf, 1st fold
outname=paste(nameid,"rf",1,"Rdata",sep=".")
load(outname)

finaltestPred <- predict(tmpfit, newdata=testtotal)

pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(finaltestPred)
