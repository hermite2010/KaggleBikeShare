## Libraries I am going to need
library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)

## Read in the data
bikeTrain <- vroom("KaggleBikeShare/train.csv")
bikeTest <- vroom("KaggleBikeShare/test.csv")

## Remove casual and registered because we can't use them to predict
bikeTrain <- bikeTrain %>%
  select(-casual, - registered)

## Cleaning & Feature Engineering
bike_recipe <- recipe(count~., data=bikeTrain) %>%
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather=factor(weather, levels=1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season=factor(season, levels=1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_time(datetime, features="hour") %>%
  step_rm(datetime)
prepped_recipe <- prep(bike_recipe)
bake(prepped_recipe, new_data = bikeTrain) #Make sure recipe work on train
bake(prepped_recipe, new_data = bikeTest) #Make sure recipe works on test


# Linear Regression -------------------------------------------------------
## Define the model
lin_model <- linear_reg() %>%
  set_engine("lm")

## Set up the whole workflow
bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(lin_model) %>%
  fit(data=bikeTrain)

## Look at the fitted LM model this way
extract_fit_engine(bike_workflow) %>%
  summary()

## Get Predictions for test set AND format for Kaggle
test_preds <- predict(bike_workflow, new_data = bikeTest) %>%
  bind_cols(., bikeTest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write prediction file to CSV
vroom_write(x=test_preds, file="KaggleBikeShare/TestPreds.csv", delim=",")

# Poisson Regression ------------------------------------------------------

pois_mod <- poisson_reg() %>% 
  set_engine("glm")

bike_pois_workflow <- workflow() %>% 
  add_recipe(bike_recipe) %>% 
  add_model(pois_mod) %>% 
  fit(data = bikeTrain)

# GETTING THE POIS PREDICTIONS
bike_predictions <- predict(bike_pois_workflow,
                            new_data = bikeTest)

## Get Predictions for test set AND format for Kaggle
test_pois_preds <- predict(bike_pois_workflow, new_data = bikeTest) %>%
  bind_cols(., bikeTest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write prediction file to CSV
vroom_write(x=test_pois_preds, file="KaggleBikeShare/PoisTestPreds.csv", delim=",")