initial_images  <- read.csv("images.csv", header = FALSE)
initial_labels <- read.csv("labels.csv", header = FALSE)

N <- nrow(initial_labels)#Number of data points

K <- max(initial_labels)#Number of labels

#splitting the data into two equal parts
images <- as.matrix(initial_images[1:(N/2),])
Labels <- initial_labels[1:(N/2),]
test_images <- as.matrix(initial_images[((N/2)+1):N,])
test_Labels <- initial_labels[((N/2)+1):N,]

N <- length(Labels) #number of training images
NT <- length(test_Labels) #number of test images

# one-of-K-encoding
labels <- matrix(0, N, K)
labels[cbind(1:N, Labels)] <- 1
test_labels <- matrix(0,NT,K)
test_labels[cbind(1:N, test_Labels)] <- 1

#initial W and W0 values
W <- as.matrix(read.csv("initial_W.csv", header = FALSE))
w0 <- as.matrix(read.csv("initial_w0.csv", header = FALSE))

#doublecheck sigmoid values
sigmoid <- function(X, w, w0){
  1/(1+exp(-((X%*%w) + t(matrix(w0,ncol=nrow(X), nrow=nrow(w0))))))
}

# define the gradient functions(lab3 code)
gradient_w <- function(X, y_truth, y_predicted) {
  t(X)%*%as.matrix(((y_truth-y_predicted)*y_predicted*(1-y_predicted)))
}

gradient_w0 <- function(y_truth, y_predicted) {
  return(colSums((y_truth-y_predicted)*y_predicted*(1-y_predicted)))
}

#iteration variables
iteration <- 1
objective_values <- c()
eta <- 0.0001
epsilon <- 1e-3
max_iteration <- 500

for(i in 1:max_iteration){
  y_predicted <- sigmoid(images,W,w0)
  
  #saving values to plot later
  objective_values <- c(objective_values, 0.5 * sum((labels-y_predicted)^2))
  w_old <- W
  w0_old <- w0
  
  #updating W and w0 with gradient
  W <- W + eta * gradient_w(images, labels, y_predicted)
  w0 <- w0 + eta * gradient_w0(labels, y_predicted)
  
  #loop exit condition
  if (sqrt(sum((w0 - w0_old)^2) + sum((W - w_old)^2)) < epsilon) {
    break
  }
}

# plot objective function during iterations
plot(1:max_iteration, objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

# calculate confusion matrix for training set
y_predicted <- sigmoid(images, W, w0)
training_labels_learned <- apply(y_predicted, 1, which.max)
confusion_matrix <- table(training_labels_learned, Labels)
print(confusion_matrix)

# calculate confusion matrix for test set
y_predicted_test <- sigmoid(test_images,W,w0)
test_labels_learned <- apply(y_predicted_test,1,which.max) 
confusion_matrix_test <- table(test_labels_learned,test_Labels)
print(confusion_matrix_test)
