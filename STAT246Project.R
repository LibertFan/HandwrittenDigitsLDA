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


## Compute Sigma
compute_sigma <- function(mu, labels, x){
	total = (x[1,]-mu[[labels[1]+1]])%*%t(x[1,]-mu[[labels[1]+1]]) # initialize
	for (i in 2:length(x[,1])){ # loop through samples
		total = total + ((x[i,]-mu[[labels[i]+1]])%*%t(x[i,]-mu[[labels[i]+1]])) # add to sum
	}
	return((1/length(x[,1]))*total) # divided by n
}


## Covariance matrix regularization and inversion method.
inv_reg_covmat <- function(sigma, lambda){
	sigma_l = (1-lambda)*sigma + (lambda/4)*diag(length(sigma[1,])) # regularize as suggested
	return(ginv(sigma_l)) # return inverse of regularized matrix
}


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
	error_rate = 1-mean((labels == classes)+0) # compute error rate
	return(list(error_rate, classes)) # return error rate and predicted classes
}


## Cross validation
lambda = seq(0.05, 0.95, 0.05) # initializing lambdas
p = rep(0.1, 10) # initializing pi
training_labels = sort(rep(seq(0,9),400)) # readjusting training labels
errors = list() # initializing list of errors
for(i in 1:length(lambda)){ # looping through all values of lambda
	error_i = c() # intialize vector of errors at this value of lambda
	for (j in 1:5){ # training and testing at this lambda 5 times
		mu = list() # initializing list of means
		cv.train.data = list(); cv.test.data = list(); cv.test.labels = list() # initializing cross-validation data lists
		for (k in 0:9){ # looping through each class
			inds = sample(500,400) # randomly sample 400
			cv.train.data[[k+1]] = training.data[which(training.label==k),][inds,]+0 # store training data
			cv.test.data[[k+1]] = training.data[which(training.label==k),][-inds,]+0 # store test data
			cv.test.labels[[k+1]] = training.label[which(training.label==k)][-inds] # store test labels
			mu[[k+1]] = colMeans(cv.train.data[[k+1]]) # calculate mean
		}
		cv.train.data.mat = cv.train.data[[1]] # initializing matrix of training data
		for(k in 2:length(cv.train.data)){ # transforming training data from list to matrix
			cv.train.data.mat = rbind(cv.train.data.mat, cv.train.data[[k]])
		}
		cv.test.data.mat = cv.test.data[[1]] # initializing matrix ot test data
		for(k in 2:length(cv.test.data)){ # transforming test data from list to matrix
			cv.test.data.mat = rbind(cv.test.data.mat, cv.test.data[[k]])
		}
		sigma = compute_sigma(mu, training_labels, cv.train.data.mat) # computing covariance matrix
		cv.test.labels = unlist(cv.test.labels) # transforming test labels from list to vector
		error_i = c(error_i, test_classifier(mu, sigma, p, lambda[i], cv.test.data.mat, cv.test.labels)[[1]]) # appending to vector of errors
	}
	errors[[i]] = error_i # appending to list of errors
}
pdf("ErrorPlot.pdf", width=13, height=8)
plot(sort(rep(lambda,5)), unlist(errors), main="Error Rate vs. Lambda", xlab="Lambda", ylab="Error Rate")
points(lambda, colMeans(matrix(unlist(errors), nrow=5, byrow=F)), pch=10, cex=5)
dev.off()
error_mat = matrix(unlist(errors), nrow=5, byrow=F)
colnames(error_mat) = lambda
error_mat
colMeans(error_mat)


## Testing final classifier
mu = list() # initialize means
p = rep(0.1,10) # initialize pi
lambda = 0.2 # set lambda based on cross-validation
for (k in 0:9){
	mu[[k+1]] = colMeans(training.data[which(training.label==k),]) # compute mean of k'th digit
}
sigma = compute_sigma(mu, training.label, training.data) # compute covariance matrix
test_results = test_classifier(mu, sigma, p, lambda, test.data, test.label) # test classifier on test data
test_results[[1]] # print error rate from test
class_error = c() # initializing class-specific error rate vector
for (i in 0:9){
	class_error = c(class_error, 1 - mean((test_results[[2]]==test.label)[test.label==i])) # appending class-specific error rate for i'th class
}
class_error # print class-specific error rate vector
pdf("ClassErrorRate.pdf", width=13, height=8)
barplot(class_error, main="Class-specific Error Rates", xlab="Class", names.arg=c("0","1","2","3","4","5","6","7","8","9"), ylab="Error Rate")
dev.off()



