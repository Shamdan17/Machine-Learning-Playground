# read data into memory
data_set <- read.csv("data_set.csv")

# get X and y values
X <- data_set$eruptions
y <- data_set$waiting

# get number of classes and number of features
D <- 1

# get train and test splits
P <- 25
set.seed(421)
train_indices <-  c(seq(1,150))
test_indices <- c(seq(151,272))
X_train <- X[train_indices]
y_train <- y[train_indices]
X_test <- X[-train_indices]
y_test <- y[-train_indices]

# get numbers of train and test samples
N_train <- length(y_train)
N_test <- length(y_test)

# create necessary data structures
node_indices <- list()
is_terminal <- c()
need_split <- c()

#node_features <- c()
node_splits <- c()
node_avgs <- c()

# put all training instances into the root node
node_indices <- list(1:N_train)
is_terminal <- c(FALSE)
need_split <- c(TRUE)

# learning algorithm
while (1) {
  # find nodes that need splitting
  split_nodes <- which(need_split)
  # check whether we reach all terminal nodes
  if (length(split_nodes) == 0) {
    break
  }
  # find best split positions for all nodes
  for (split_node in split_nodes) {
    data_indices <- node_indices[[split_node]]
    need_split[split_node] <- FALSE
    # check whether node has less than p elements
    if (length(data_indices) < P) {
      is_terminal[split_node] <- TRUE
    } else {
      is_terminal[split_node] <- FALSE
      
      unique_values <- sort(unique(X_train[data_indices]))
      split_positions <- (unique_values[-1] + unique_values[-length(unique_values)]) / 2
      split_scores <- rep(0, length(split_positions))
      for (s in 1:length(split_positions)) {
        left_indices <- data_indices[which(X_train[data_indices] < split_positions[s])]
        right_indices <- data_indices[which(X_train[data_indices] >= split_positions[s])]
        split_scores[s] <- (1 / length(data_indices)) * sum((y_train[left_indices] - mean(y_train[left_indices]))^2, na.rm = TRUE) +
                           (1 / length(data_indices)) * sum((y_train[right_indices] - mean(y_train[right_indices]))^2, na.rm = TRUE)
      }
      best_score <- min(split_scores)
      best_split <- split_positions[which.min(split_scores)]
      
      # decide where to split on which feature
      node_splits[split_node] <- best_split
      
      # create left node using the selected split
      left_indices <- data_indices[which(X_train[data_indices] < best_split)]
      node_indices[[2 * split_node]] <- left_indices
      is_terminal[2 * split_node] <- FALSE
      need_split[2 * split_node] <- TRUE
      
      # create left node using the selected split
      right_indices <- data_indices[which(X_train[data_indices] >= best_split)]
      node_indices[[2 * split_node + 1]] <- right_indices
      is_terminal[2 * split_node + 1] <- FALSE
      need_split[2 * split_node + 1] <- TRUE
    }
  }
}

assignVal <- function(x){
  idx <- 1
  while(1){
    if(is_terminal[idx]){
      return(mean(y_train[node_indices[[idx]]]))
    }else{
      if(x <= node_splits[idx]){
        idx <- 2 * idx
      }else{
        idx <- 2 * idx + 1
      }
    }
  }
}


N_test <- length(X_test)
# traverse tree for test data points
y_predicted <- sapply(X_test, assignVal)

assignVal(X_test[18])

RMSE <- function(pred, actual){
  return(sqrt(mean((pred-actual)^2)))
}


print(sprintf("RMSE is %g when P = %g",RMSE(y_predicted, y_test),P))

#Plot the points
data_pts <- seq(1.4,5.2,0.01)
data_vals <- lapply(data_pts,FUN=assignVal)
plot(data_pts, data_vals,
     type = "l", lwd = 2, las = 1, main = "P = 25",
     xlab = "Eruption time (min)", ylab = "Waiting time to next eruption (min)")
points(X_train, y_train, type = "p", pch = 19, col = "blue")
points(X_test, y_test, type = "p", pch = 19, col = "red")
legend(1.25, 98, legend=c("training", "test"),
       col=c("blue","red"),  pch=19)



calcRMSEforP <- function(p){
  node_indices <- list()
  is_terminal <- c()
  need_split <- c()
  
  node_splits <- c()
  node_avgs <- c()
  
  # put all training instances into the root node
  node_indices <- list(1:N_train)
  is_terminal <- c(FALSE)
  need_split <- c(TRUE)
  
  
  # learning algorithm
  while (1) {
    # find nodes that need splitting
    split_nodes <- which(need_split)
    # check whether we reach all terminal nodes
    if (length(split_nodes) == 0) {
      break
    }
    # find best split positions for all nodes
    for (split_node in split_nodes) {
      data_indices <- node_indices[[split_node]]
      need_split[split_node] <- FALSE
      # check whether node has less than p elements
      if (length(data_indices) <= p) {
        is_terminal[split_node] <- TRUE
      } else {
        is_terminal[split_node] <- FALSE
        
        unique_values <- sort(unique(X_train[data_indices]))
        split_positions <- (unique_values[-1] + unique_values[-length(unique_values)]) / 2
        split_scores <- rep(0, length(split_positions))
        for (s in 1:length(split_positions)) {
          left_indices <- data_indices[which(X_train[data_indices] < split_positions[s])]
          right_indices <- data_indices[which(X_train[data_indices] >= split_positions[s])]
          split_scores[s] <- (1 / length(data_indices)) * 
            (sum((y_train[left_indices] - mean(y_train[left_indices]))^2, na.rm = TRUE) +
            sum((y_train[right_indices] - mean(y_train[right_indices]))^2, na.rm = TRUE))
        }
        best_score <- min(split_scores)
        best_split <- split_positions[which.min(split_scores)]
        
        # decide where to split on which feature
        node_splits[split_node] <- best_split
        
        # create left node using the selected split
        left_indices <- data_indices[which(X_train[data_indices] < best_split)]
        node_indices[[2 * split_node]] <- left_indices
        is_terminal[2 * split_node] <- FALSE
        need_split[2 * split_node] <- TRUE
        
        # create right node using the selected split
        right_indices <- data_indices[which(X_train[data_indices] >= best_split)]
        node_indices[[2 * split_node + 1]] <- right_indices
        is_terminal[2 * split_node + 1] <- FALSE
        need_split[2 * split_node + 1] <- TRUE
      }
    }
  }
  
  assignVal <- function(x){
    idx <- 1
    while(1){
      if(is_terminal[idx]){
        return(mean(y_train[node_indices[[idx]]]))
      }else{
        if(x <= node_splits[idx]){
          idx <- 2 * idx
        }else{
          idx <- 2 * idx + 1
        }
      }
    }
  }
  
  y_predicted <- sapply(X_test, assignVal)
  
  return(RMSE(y_predicted, y_test))
}

data_pts = seq(5, 50, 5)
data_vals <- sapply(data_pts, calcRMSEforP)

plot(data_pts, data_vals,
     type = "l", lwd = 2, las = 1,
     xlab = "Pre-Pruning size (P)", ylab = "RMSE")
points(data_pts, data_vals, type = "p", pch = 19, col = "black")

