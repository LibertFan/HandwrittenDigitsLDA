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
lambda = seq(0.05, 0.95, 0.05)
p = rep(0.1, 10)
errors = list()
for(i in 1:length(lambda)){
	error_i = c()
	for (j in 1:5){
		mu = list()
		cv.train.data = list()
		cv.test.data = list()
		cv.test.labels = list()
		for (k in 0:9){
			inds = sample(500,400)
			cv.train.data[[k+1]] = training.data[which(training.label==k),][inds,]+0
			cv.test.data[[k+1]] = training.data[which(training.label==k),][-inds,]+0
			cv.test.labels[[k+1]] = training.label[which(training.label==k)][-inds]
			mu[[k+1]] = colMeans(cv.train.data[[k+1]])
		}
		sigma = cov(matrix(unlist(cv.train.data), nrow=4000, byrow=T))
		cv.test.data = matrix(unlist(cv.test.data), nrow=1000, byrow=T)
		cv.test.labels = unlist(cv.test.labels)		
		error_i = c(error_i, test_classifier(mu, sigma, p, lambda[i], cv.test.data, cv.test.labels)[[1]])
	}
	errors[[i]] = error_i
}
pdf("ErrorPlot.pdf", width=13, height=8)
plot(sort(rep(lambda,5)), unlist(errors), main="Error Rate vs. Lambda", xlab="Lambda", ylab="Error Rate")
points(lambda, colMeans(matrix(unlist(errors), nrow=5, byrow=F)), pch=10, cex=5)
dev.off()
error_mat = matrix(unlist(errors), nrow=5, byrow=F)
colnames(error_mat) = lambda
error_mat
colMeans(error_mat)





## Bayes Classification Method
### Returns the predicted digit based on Bayes classification as specified in Part a.
### Requires mu as a list of vectors, sigma_inv as a matrix, p as a vector, and x (data) as a vector.
bayes_classify <- function(mu, sigma_inv, p, x){
	w_0 = c(); w=list(); vals = c() # initializing
	for (i in 1:10){
		w_0 = c(w_0, -(1/2)*(mu[[i]] %*% sigma_inv %*% mu[[i]]) + log(p[i])) # intercept for i'th class
		w[[i+1]] = sigma_inv %*% mu[[i]] # slope for i'th class
		vals = c(vals, t(w[[i+1]]) %*% x + w_0[i]) # storing w_0i + w_i*x
	}
	return(which(vals==max(vals))-1) # returning class which maximizes w_0i + w_i*x
}


## Test classifier
### Returns error rate and predicted classes for a set of test data.
### Requires mu as a list of vectors, sigma as a matrix, p as a vector, lambda as a number, x as a matrix, and labels as a vector.
test_classifier <- function(mu, sigma, p, lambda, x, labels){
	sig_inv = inv_reg_covmat(sigma, lambda) # regularize and invert covariance matrix
	classes = c() # initialize
	for (i in 1:length(x[,1])){ # loop through examples in test set x
		classes = c(classes, bayes_classify(mu, sig_inv, p, x[i,])) # append to vector of predicted classes
	}
	error_rate = mean((labels == classes)+0) # compute error rate
	return(list(error_rate, classes)) # return error rate and predicted classes
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

