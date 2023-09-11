##
## Bike Share EDA Code
##

## Libraries
library(tidyverse)
library(vroom)
library(patchwork)

## Read in the Data
bike <- vroom("./KaggleBikeShare/train.csv")

dplyr::glimpse(bike)
skimr::skim(bike)
DataExplorer::plot_intro(bike)
DataExplorer::plot_correlation(bike)
DataExplorer::plot_bar(bike)
DataExplorer::plot_histogram(bike)
DataExplorer::plot_missing(bike)
GGally::ggpairs(bike)

scat_1 <- ggplot(bike, mapping = aes(x = windspeed, y = count))+
  geom_point()+
  geom_smooth(se = FALSE)+
  labs(title = "Windspeed and Count")
scat_2 <- ggplot(bike, mapping = aes(x = temp, y = count))+
  geom_point()+
  geom_smooth(se=FALSE)+
  labs(title = "Temperature and Count")

hw_2 <- (DataExplorer::plot_correlation(bike) + DataExplorer::plot_missing(bike)) / (scat_2 + scat_1)

hw_2


