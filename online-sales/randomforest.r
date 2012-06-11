library('randomForest')

train <- read.csv('TrainingDataset.csv')
test <- read.csv('TestDataset.csv')

appendNAs <- function(dataset, cols) {
  append_these = data.frame( is.na(dataset[, cols] ))
  names(append_these) = paste(names(append_these), "NA", sep = "_")
  dataset = cbind(dataset, append_these)
  dataset[is.na(dataset)] = -1
  return(dataset)
}

train <- appendNAs(train, 13:ncol(train))
test <- appendNAs(test, 2:ncol(test))

submission <- data.frame(id = test$id)
for (i in 1:12) {
  rf = randomForest(train[,13:ncol(train)], train[,i], do.trace=TRUE, importance=TRUE, sampsize=nrow(train)*.8, ntree=500)
  submission[,1+i] = predict(rf, test[,2:ncol(test)])
}
names(submission) <- c('id', names(train[,1:12]))
write.csv(submission, file="submission.csv", row.names=FALSE)
