library(keras)
install_keras()
library(tfruns)
library(rbenchmark)


library(reticulate)
library(usethis)
library(devtools)

install.packages('keras') 
install.packages('tensorflow')
library(keras) 
library(tensorflow) 
install_keras() 
install_tensorflow() 

##test 
library(tensorflow)
tf$constant("Hellow Tensorflow")

# Hyperparameter flags ---------------------------------------------------

FLAGS <- flags(
  flag_numeric("dense1", 64),
  flag_numeric("dense2", 64),
  flag_numeric("dense3", 64),
  flag_numeric("seed", 22071997)
)

# Data Preparation ---------------------------------------------------

set.seed(FLAGS$seed)
getwd()
setwd("C:/Users/Lisa/Documents/M1_SSD/Projet_tut/Neural_Network")
data <- read.csv2("tableau_FINAL.csv")


# data <- na.omit(data)

# The data, shuffled and split between train and test sets
labels <- data[,1]
data <- data[,-1]
train_i <- sample(nrow(data), nrow(data)*80/100)

row.names(data) <- 1:nrow(data)

test_i <- as.numeric(row.names(data[-train_i,]))

train_data <- data[train_i,]
train_labels <- labels[train_i]
test_data <- data[test_i,]
test_labels <- labels[test_i]



row.names(train_data) <- 1:length(train_data[,1])
row.names(test_data) <- 1:length(test_data[,1])




# Test data is *not* used when calculating the mean and std.

# Normalize training data
train_data <- scale(train_data) 

# Use means and standard deviations from training set to normalize test set
col_means_train <- attr(train_data, "scaled:center") 
col_stddevs_train <- attr(train_data, "scaled:scale")
test_data <- scale(test_data, center = col_means_train, scale = col_stddevs_train)



# Display training progress by printing a single dot for each completed epoch.
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    cat("\repoch : ", epoch, "/", epochs, " (", sprintf("%.2f",epoch*100/epochs), "%)   ")
  }
)    



# Define Model --------------------------------------------------------------


epochs <- 1000


build_model <- function() {
  model <- keras_model_sequential() %>%
    layer_dense(units = FLAGS$dense1, activation = "relu",
                input_shape = dim(train_data)[2]) %>%
    layer_dense(units = FLAGS$dense2, activation = "relu") %>%
    layer_dense(units = FLAGS$dense3, activation = "relu") %>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(),
    metrics = list("mean_absolute_error")
  )
  
  model
}


# The patience parameter is the amount of epochs to check for improvement.

early_stop <- callback_early_stopping(monitor = "val_loss", patience = 30)



# Building the model

model <- build_model()

# Training & Evaluation ----------------------------------------------------

history <- model %>% fit(
  train_data,
  train_labels,
  epochs = epochs,
  # view_metrics =TRUE,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(print_dot_callback, early_stop)
)

plot(history, metrics = "mean_absolute_error", smooth = FALSE)

score <- model %>% evaluate(
  test_data, test_labels,
  verbose = 0
)

cat('Test loss:', score$loss, '\n')
cat('Test accuracy:', score$mean_absolute_error, '\n')


test_predictions <- model %>% predict(test_data)
plot(test_predictions,test_labels, main = paste("Mean absolute error on test set :", round(score$mean_absolute_error,2)), pch=20, cex=0.5)
abline(0,1)




