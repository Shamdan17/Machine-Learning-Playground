safelog <- function(x) {
  return (log(x + 1e-100))
}

initial_images  <- read.csv("images.csv", header = FALSE)
initial_labels <- read.csv("labels.csv", header = FALSE)

N <- nrow(initial_labels)#Number of data points
K <- max(initial_labels)#Number of labels

#splitting the data into two equal parts
images <- initial_images[1:(N/2),]
labels <- initial_labels[1:(N/2),]
test_images <- initial_images[((N/2)+1):N,]
test_labels <- initial_labels[((N/2)+1):N,]

#Take the means of the columns for each label
class_means <- sapply(X = 1:K, function(c){colMeans(images[labels==c,])})
head(class_means)

#Take the variance of the columns for each label
variances <- sapply(X = 1:K, function(c){apply(images[labels==c,], 2, function(y){sum((y - mean(y))^2)/(length(y))})})
head(variances)

#Take the std_dev of the each label
std_devs <- sqrt(variances)
head(std_devs)

#prior probability 
priors <- sapply(X = 1:K, function(c){mean(labels == c)})
head(priors)

#Since the Sigma matrix is a diagonal matrix, x(transpose) * Sigma(inv) * x 
#will be equal to the sum of each element in x^2 multiplied by the corresponding 
#diagonal element in Sigma(inv)

#Moreover, since Sigma is a diagonal matrix, its inverse is just another diagonal
#matrix with each of the diagonal elements being the reciprocal of the corresponding
#elements in Sigma

g <- function(image){apply(image, 1, function(inp){sapply(X= 1:K, function(c){
  -0.5 * sum((inp - class_means[,c])^2 * (1/variances[,c])) + log(priors[c]) - 0.5 * sum(safelog(variances[,c]))
})})}


#classifying the learning image set
learning_set_scores <- g(images)
learning_set_labels <- apply(learning_set_scores, 2, which.max)
#confusion matrix
learning_confusion_matrix <- table(labels, learning_set_labels)

head(learning_confusion_matrix)

#classifying the test image set
test_set_scores <- g(test_images)
test_set_labels <- apply(test_set_scores, 2, which.max)
#confusion matrix
test_confusion_matrix <- table(test_labels, test_set_labels)

head(test_confusion_matrix)
