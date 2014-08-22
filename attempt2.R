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
nameid <- "att2"
compute <- TRUE
numdata <- 19000
numvalid <- 620
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


# inform
    print(paste(i,  testacclist$rf[i]))
    print(paste(i,  validacclist$rf[i]))

}

# compute average and standard deviation of testing and validation accuracies

acc$avgtestacc <- c(0, 0, mean(testacclist$rf), 0)
acc$sdtestacc <- c(0, 0, sd(testacclist$rf), 0)


# perform validation using the best fit of each method

bestrf <- which.max(testacclist$rf)

bests<-c(cart=0, bcart=0, rf=bestrf, gbm=0)
acc$avgvalidacc <- c(0, 0, validacclist$rf[bestrf], 0)

print(acc)

dput(acc, paste(nameid,"acc.R",sep="."))
dput(bests, paste(nameid,"bests.R",sep="."))

# predict testing data

# best model: rf, 1st fold
outname=paste(nameid,"rf",8,"Rdata",sep=".")
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
