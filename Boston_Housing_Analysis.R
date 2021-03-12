library(data.table)
library(tidyverse)
library(inspectdf)
library(magrittr)


dataset <- fread("BostonHausing.csv")


dataset %>% glimpse()

dataset %>% inspect_na()


dataset %>% colnames()

dataset$chas

dataset$chas %>% table() %>% prop.table()

dataset %>% dim()

dataset$chas <- dataset$chas %>% as.factor()

library(caret)
id <- createDataPartition(y = dataset$chas,p = 0.8,list = FALSE)

dataset_train <- dataset[id,]
dataset_test <- dataset[-id,]


library(rJava)

Sys.setenv(JAVA_HOME= "C:\\Program Files\\Java\\jre1.8.0_271")
Sys.getenv("JAVA_HOME")



library(h2o)

h2o.init(nthreads = -1, max_mem_size = '2g', ip = "127.0.0.1", port = 54321)

train <- as.h2o(dataset_train)
test <- as.h2o(dataset_test)

#h2o_data <- dataset %>% as.h2o()

#h2o_data <- h2o_data %>% h2o.splitFrame(ratios = 0.8, seed = 123)
#train <- h2o_data[[1]]
#test <- h2o_data[[2]]

target <- 'chas'
features <- dataset %>% select(-chas) %>% names()


#TRAIN MODEL


pure_logistic <- h2o.glm(family= "binomial", 
                         x = features, 
                         y = target, 
                         lambda = 0, 
                         training_frame = train)


show_coeffs <- function(model_selected) {
  model_selected@model$coefficients_table %>% 
    as.data.frame() %>% 
    mutate_if(is.numeric, function(x) {round(x, 3)}) %>% 
    filter(coefficients != 0) %>% 
    knitr::kable()
}

library(dplyr)
  
show_coeffs(pure_logistic)


lasso_logistic <- h2o.glm(family = "binomial", 
                          alpha = 1,
                          seed = 1988, 
                          x = features, 
                          y = target, 
                          training_frame = train)

show_coeffs(lasso_logistic)


ridge_logistic <- h2o.glm(family = "binomial", 
                          alpha = 0,
                          seed = 1988, 
                          x = features, 
                          y = target, 
                          training_frame = train)

show_coeffs(ridge_logistic)


my_cm <- function(model_selected) {
  pred <- h2o.predict(model_selected, test) %>% 
    as.data.frame() %>% 
    pull(1)
  confusionMatrix(pred, dataset_test$chas, positive = "1") %>% 
    return()
}

lapply(list(pure_logistic, lasso_logistic, ridge_logistic), my_cm)

#my_cm(lasso_logistic)


