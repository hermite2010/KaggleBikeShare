## Libraries I am going to need
library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)
library(rpart)
library(ranger)
library(stacks)

## Read in the data
bikeTrain <- vroom("KaggleBikeShare/train.csv")
bikeTest <- vroom("KaggleBikeShare/test.csv")

## Remove casual and registered because we can't use them to predict
bikeTrain <- bikeTrain %>%
  select(-casual, - registered)

## Cleaning & Feature Engineering
bike_recipe <- recipe(count~., data=bikeTrain) %>%
#  step_mutate(count = log(as.numeric(count))) %>%  #Attempted log transformation
#  step_log() %>% # Another attempted log transformation
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Relabel weather 4 to 3
  step_mutate(weather=factor(weather, levels=1:3, labels=c("Sunny", "Mist", "Rain"))) %>%
  step_mutate(season=factor(season, levels=1:4, labels=c("Spring", "Summer", "Fall", "Winter"))) %>%
  step_mutate(holiday=factor(holiday, levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_mutate(workingday=factor(workingday,levels=c(0,1), labels=c("No", "Yes"))) %>%
  step_time(datetime, features="hour") %>%
  step_rm(datetime) %>% 
  step_dummy(all_nominal_predictors()) %>% #make dummy variables FOR PENALIZED
  step_normalize(all_numeric_predictors()) # Make mean 0, sd=1 FOR PENALIZED
prepped_recipe <- prep(bike_recipe)
bake(prepped_recipe, new_data = bikeTrain) #Make sure recipe work on train
bake(prepped_recipe, new_data = bikeTest) #Make sure recipe works on test

# Log Transformation ------------------------------------------------------
## Transform to log(count) - I can only do this on train set because
## test set does not have count.  Hence, I am doing this outside of recipe
## because I only apply this to the train set
logTrainSet <- bikeTrain %>%
  mutate(count=log(count))
## Define the model
lin_model <- linear_reg() %>%
  set_engine("lm")
## Set up the whole workflow
log_lin_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(lin_model) %>%
  fit(data=logTrainSet) #Make sure to use the log(count) dataset
## Get Predictions for test set AND format for Kaggle
log_lin_preds <- predict(log_lin_workflow, new_data = bikeTest) %>% #This predicts log(count)
  mutate(.pred=exp(.pred)) %>% # Back-transform the log to original scale
  bind_cols(., bikeTest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle
## Write predictions to CSV
vroom_write(x=log_lin_preds, file="./LogLinearPreds.csv", delim=",")

# Linear Regression -------------------------------------------------------
## Define the model
lin_model <- linear_reg() %>%
  set_engine("lm")

## Set up the whole workflow
lin_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(lin_model) %>%
  fit(data=bikeTrain)

## Look at the fitted LM model this way
extract_fit_engine(lin_workflow) %>%
  summary()

## Get Predictions for test set AND format for Kaggle
test_preds <- predict(lin_workflow, new_data = bikeTest) %>%
  bind_cols(., bikeTest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
#  mutate(count=exp(count)) %>% #attempt to transform back to original scale (0, prediction)
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
#  mutate(count = exp(count)) %>%  # attempt to rescale the count variable
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle

## Write prediction file to CSV
vroom_write(x=test_pois_preds, file="KaggleBikeShare/PoisTestPreds.csv", delim=",")

# Penalized Regression ----------------------------------------------------

# For LOG TRANSFORMATION

logTrainSet <- bikeTrain %>%
  mutate(count=log(count))

## Penalized regression model
preg_model <- linear_reg(penalty=0, mixture=0) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

preg_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(preg_model) %>%
  fit(data=logTrainSet) # USING THE LOG TRAINING SET

test_penal_preds <- predict(preg_wf, new_data=bikeTest) %>%
  mutate(.pred=exp(.pred)) %>% # RESETS THE LOG TRANSFORMATION
  bind_cols(., bikeTest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=test_penal_preds, file="KaggleBikeShare/PenalizedTestPreds.csv", delim=",")

# Cross Validation --------------------------------------------------------

# For LOG TRANSFORMATION

logTrainSet <- bikeTrain %>%
  mutate(count=log(count))

## Penalized regression model
preg_model <- linear_reg(penalty=tune(), 
                         mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

preg_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(preg_model)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 100)

folds <- vfold_cv(logTrainSet, v = 10, repeats=1)

## Run the CV
CV_results <- preg_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae, rsq)) #Or leave metrics NULL

## Plot Results (example)
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

## Find Best Tuning Parameters
bestTune <- CV_results %>%
select_best("rmse")

## Finalize the Workflow & fit it
final_wf <- preg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=logTrainSet)

## Predict
test_cv_preds <- predict(final_wf, new_data=bikeTest) %>%
  mutate(.pred=exp(.pred)) %>% # RESETS THE LOG TRANSFORMATION
  bind_cols(., bikeTest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=test_cv_preds, file="KaggleBikeShare/CVTestPreds.csv", delim=",")


# Decision Tree -----------------------------------------------------------

tree_mod <- decision_tree(tree_depth = tune(),
                          cost_complexity = tune(),
                          min_n = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

logTrainSet <- bikeTrain %>%
  mutate(count=log(count))

## Workflow and model and recipe

tree_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(tree_mod)

## set up grid of tuning values

tuning_grid <- grid_regular(tree_depth(),
                            cost_complexity(),
                            min_n(),
                            levels = 5)

## set up k-fold CV

folds <- vfold_cv(logTrainSet, v = 5, repeats=1)

## Run the CV

CV_results <- tree_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae, rsq)) #Or leave metrics NULL

## find best tuning parameters

bestTune <- CV_results %>%
  select_best("rmse")

## Finalize workflow and prediction 

final_wf <- tree_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=logTrainSet)

# Predict and format for Kaggle

test_tree_preds <- predict(final_wf, new_data=bikeTest) %>%
  mutate(.pred=exp(.pred)) %>% # RESETS THE LOG TRANSFORMATION
  bind_cols(., bikeTest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=test_tree_preds, file="KaggleBikeShare/TreeTestPreds.csv", delim=",")


# Random Forest -----------------------------------------------------------
rand_for_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=750) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("regression")

logTrainSet <- bikeTrain %>%
  mutate(count=log(count))

## Workflow and model and recipe

rand_for_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(rand_for_mod)

## set up grid of tuning values

rand_tuning_grid <- grid_regular(mtry(range = c(1,(ncol(logTrainSet)))),
                            min_n(),
                            levels = 5)

## set up k-fold CV

rand_folds <- vfold_cv(logTrainSet, v = 5, repeats=1)

## Run the CV

CV_results <- rand_for_wf %>%
  tune_grid(resamples=rand_folds,
            grid=rand_tuning_grid,
            metrics=metric_set(rmse, mae, rsq)) #Or leave metrics NULL

## find best tuning parameters

bestTune <- CV_results %>%
  select_best("rmse")

## Finalize workflow and prediction 

final_wf <- rand_for_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=logTrainSet)

# Predict and format for Kaggle

test_rand_preds <- predict(final_wf, new_data=bikeTest) %>%
  mutate(.pred=exp(.pred)) %>% # RESETS THE LOG TRANSFORMATION
  bind_cols(., bikeTest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=test_rand_preds, file="KaggleBikeShare/RandForestTestPreds.csv", delim=",")

# Stacking Models ---------------------------------------------------------

# Using the log of the data
logTrainSet <- bikeTrain %>%
  mutate(count=log(count))

# Splitting the data for Cross Validation
folds <- vfold_cv(logTrainSet, v = 5, repeats = 1)

# Control Settings for stacking models
untunedModel <- control_stack_grid()
tunedModel <- control_stack_resamples()

# Penalized regression model
preg_model <- linear_reg(penalty=tune(), 
                         mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

preg_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(preg_model)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

# RUNNING THE CROSS VALIDATION
preg_models <- preg_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(rmse, mae, rsq),
            control = untunedModel) # THIS IS WHERE WE START STACKING

# RANDOM FOREST 
rand_for_mod <- rand_forest(mtry = tune(),
                            min_n = tune(),
                            trees=750) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("regression")

## Workflow and model and recipe
rand_for_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(rand_for_mod)

## set up grid of tuning values
rand_tuning_grid <- grid_regular(mtry(range = c(1,(ncol(logTrainSet)))),
                                 min_n(),
                                 levels = 5)

## Running the Cross validation
rand_models <- rand_for_wf %>%
  tune_grid(resamples=folds,
            grid=rand_tuning_grid,
            metrics=metric_set(rmse, mae, rsq), #Or leave metrics NULL
            control = untunedModel)

# Specify with models to include
my_first_stack <- stacks() %>% 
  add_candidates(preg_models) %>% 
  add_candidates(rand_models)

# Fitting the stacked model
stack_mod <- my_first_stack %>% 
  blend_predictions() %>% 
  fit_members()

stackData <- as_tibble(my_first_stack)

stacked_preds <- stack_mod %>% 
  predict(new_data = bikeTest) %>% 
  mutate(.pred=exp(.pred)) %>% # RESETS THE LOG TRANSFORMATION
  bind_cols(., bikeTest) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=stacked_preds, file="KaggleBikeShare/StackedPreds.csv", delim=",")
