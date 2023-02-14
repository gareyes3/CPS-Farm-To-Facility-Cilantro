library(tidyverse)
library(pscl)
library(pROC)
library(ResourceSelection)
library(glmtoolbox)

TD<-read_csv("Product_Testing_Data.csv")
TD<-TD[,2:3]

model<-glm(Results~Cont, data = TD,family = binomial(link = "logit"))

hoslem.test(model$y, fitted(model), g = 10)


#water testing
TD_W<-read_csv("Water_Testing_Data.csv")
TD_W<-TD_W[,2:3]

model_W<-glm(Results~Cont, data = TD_W,family = binomial(link = "logit"))

hoslem.test(model_W$y, fitted(model_W), g = 10)