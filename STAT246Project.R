## Kevin O'Connor
## STAT 24610
## Final Project

## Loading data, setting up for use in R classification packages.
setwd("/Users/kevinoconnor/Documents/School/STAT246/")
library(MASS)
load('digits.RData')
num.class <- dim(training.data)[1] # Number of classes
num.training <- dim(training.data)[2] # Number of training data per class
d <- prod(dim(training.data)[3:4]) # Dimension of each training image (r*c)
num.test <- dim(test.data)[2] # Number of test data
dim(training.data) <- c(num.class*num.training, d) # Reshape training data to 2-dim matrix
dim(test.data) <- c(num.class*num.test, d) # Same for test
training.label <- rep(0:9, num.training) # Labels of training data
test.label <- rep(0:9, num.test) # Labels of test data


## Cross validation
mu = list()
cv.train.data = list()
cv.test.data = list()
cv.test.labels = list()
for (i in 0:9){
	inds = sample(500,400)
	cv.train.data[[i+1]] = training.data[which(training.label==i),][inds,]+0
	cv.test.data[[i+1]] = training.data[which(training.label==i),][-inds,]+0
	cv.test.labels[[i+1]] = training.label[which(training.label==i)][-inds]
	mu[[i+1]] = colMeans(cv.train.data[[i+1]])
}
sigma = cov(matrix(unlist(cv.train.data), nrow=4000, byrow=T))
cv.test.data = matrix(unlist(cv.test.data), nrow=1000, byrow=T)
cv.test.labels = unlist(cv.test.labels)
lambda = seq(0.05, 0.95, 0.05)
p = rep(0.1, 10)

test_classifier(mu, sigma, p, 0.3, cv.test.data, cv.test.labels)


## Bayes Classification Method
### Returns the index (digit + 1) of the class based on Bayes classification as specified in Part a.
bayes_classify <- function(mu, sigma_inv, p, x){
	w_0 = c(); vals = c() # initializing
	w = list()
	for (i in 1:10){
		w_0 = c(w_0, -(1/2)*(mu[[i]] %*% sigma_inv %*% mu[[i]]) + log(p[i])) # intercept for i'th class
		w[[i+1]] = sigma_inv %*% mu[[i]] # slope for i'th class
		vals = c(vals, t(w[[i+1]]) %*% x + w_0[i]) # storing w_0i + w_i*x
	}
	return(which(vals==max(vals))-1) # returning class which maximizes w_0i + w_i*x
}


## Test classifier
test_classifier <- function(mu, sigma, p, lambda, x, labels){
	sig_inv = inv_reg_covmat(sigma, lambda)
	classes = c()
	for (i in 1:length(x[,1])){
		classes = c(classes, bayes_classify(mu, sig_inv, p, x[i,]))
	}
	error_rate = mean((labels == classes)+0)
	return(list(error_rate, classes))
}


## Covariance matrix regularization and inversion method.
inv_reg_covmat <- function(sigma, lambda){
	sigma_l = (1-lambda)*sigma + (lambda/4)*diag(length(sigma[1,])) # regularize as suggested
	return(ginv(sigma_l)) # return inverse of regularized matrix
}






## Implementing LDA
library(MASS)
library(caret)
library(lattice)
library(hda)
train.lda <- lda(training.data*1, training.label) # Attempting LDA on the training data
good.indices = c(27:30,32,33,35:36,45:57,64:79,82,84:99,102:120,122:140,142:160,162:180,182:200,202:220,222:240,262:280,282:300,302:319,323:339,343:358,365,378,388) # Listing only non-constant variables
training.data.clipped <- training.data[,good.indices] # Removing constant variables from data set
train.lda <- lda(training.data.clipped*1, training.label) # Attempting LDA on clipped training data
train.cor <- cor(training.data.clipped*1) # Calculating correlation of remaining variables
train.dissimilarity <- 1 - abs(train.cor) # Creating dissimilarity measure
distance <- as.dist(train.dissimilarity) # Using dissimilarity as distance measure
clust <- hclust(distance) # Performing hierarchical clustering on variables
cluster.vec <- cutree(hc, h=0.10) # Create an index of clusters whose constituents have abs(correlation) > 0.9
print(cluster.vec)
training.data.clipped <- training.data.clipped[,-c(278,6)] # Removing collinear variables from training data
train.lda <- lda(training.data.clipped*1, training.label) # Attempting LDA on modified training data
test.data.clipped = test.data[,good.indices] # Removing constant variance variables from test
test.data.clipped = test.data.clipped[,-c(278,6)] # Removing collinear variables from test
test.prediction = predict(train.lda,newdata=test.data.clipped*1) # Performing prediction on test data
confusionMatrix(test.prediction$class, test.label)
pdf("ProjectLDAPlot1.pdf", width=13, height=10)
par(mar=c(1,1,1,1))
plot(train.lda, dimen=1, type="both", main="LDA Class Histograms")
dev.off()
train.hda <- hda(training.data[1:100,]*1, training.label[1:100], reg.gamm=0.75, reg.lamb=0, noutit=2)

