library(tidyverse)
library(pscl)
library(pROC)
library(ResourceSelection)
library(glmtoolbox)

data_predict = data.frame("Cont" = 1:200,
                          "Prob" = "")

TD<-read_csv("Product_Testing_Data.csv")
TD<-TD[,2:3]

model<-glm(Results~Cont, data = TD,family = binomial(link = "logit"))

(exp(-3.4393+0.4566*0))/(1-exp(-3.4393+0.4566*0))

predict(model,data_predict )

hoslem.test(model$y, fitted(model), g = 10)


#water testing
TD_W<-read_csv("Water_Testing_Data.csv")
TD_W$Results<-as.factor(TD_W$Results)
TD_W<-TD_W[,2:3]

model_W<-glm(Results~Cont, data = TD_W,family = binomial(link = "logit"))

hoslem.test(model_W$y, fitted(model_W), g = 10)

(exp(-19.8+3.424*0))/(1-exp(-19.8+3.424  *0))



