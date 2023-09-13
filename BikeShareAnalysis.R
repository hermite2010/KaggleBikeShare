## Libraries
library(tidyverse)
library(vroom)
library(tidymodels)

# Data
bike_test <- vroom("./KaggleBikeShare/test.csv")
bike_train <- vroom("./KaggleBikeShare/train.csv")

# Cleaning Section
bike_train <- bike_train %>% 
  mutate(workingday = as.factor(workingday)) %>%
  select(-casual, -registered)
# Made the workingday variable a factor instead of a numeric

# Feature Engineer Section
my_recipe <- recipe(count ~ ., data=bike_train) %>% # Set model formula and d
  step_mutate(temp_diff = (atemp - temp)) %>% # added the registered and causal to get total users
  step_corr(all_numeric_predictors(), threshold = 0.9) # removes variables that have correlation less than 0.5
prepped_recipe <- prep(my_recipe)
head(bake(prepped_recipe, new_data=bike_test), n = 15)
