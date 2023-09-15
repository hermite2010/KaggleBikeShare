## Libraries
library(tidyverse)
library(vroom)
library(tidymodels)

# Data
bike_test <- vroom("./KaggleBikeShare/test.csv")
bike_train <- vroom("./KaggleBikeShare/train.csv")

# Cleaning Section
bike_train <- bike_train %>% 
#  mutate(workingday = as.factor(workingday)) %>% # Made the workingday variable a factor instead of a numeric
  mutate(weather = ifelse(weather == 4, 3, weather)) %>% 
  select(-casual, -registered) # Removing the casual and registered variables


# Feature Engineer Section
my_recipe <- recipe(count ~ ., data=bike_train) %>% # Set model formula and d
  step_mutate(temp_diff = (atemp - temp)) %>% # added difference between actual and felt temperatures
  step_corr(all_numeric_predictors(), threshold = 0.9) # removes variables that have correlation less than 0.5
prepped_recipe <- prep(my_recipe)
head(bake(prepped_recipe, new_data=bike_test), n = 15)

# Linear Regression
my_mod <- linear_reg() %>% 
  set_engine("lm")

bike_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(my_mod) %>% 
  fit(data = bike_train)

bike_predictions <- predict(bike_workflow,
                            new_data=bike_test)



bike_predictions <- bike_predictions %>% 
  mutate(datetime = bike_test$datetime,
         count = ifelse(.pred<0,0,.pred)) %>% 
  select(datetime, count)

view(bike_predictions)

bike_predictions$datetime <- as.character(format(bike_predictions$datetime))

vroom_write(bike_predictions, file = "my_first_submission.csv", delim = ",")

write_csv(bike_predictions, file = "my_first_submission.csv")

