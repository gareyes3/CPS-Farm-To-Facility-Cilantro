library(tidyverse)
library(pscl)
library(pROC)
library(ResourceSelection)
library(glmtoolbox)
library(fmsb)
library(glmnet)

data_predict = data.frame("Cont" = 1:200,
                          "Prob" = "")

TD<-read_csv("Data/Product_Testing_Data.csv")
TD<-TD[,2:3]

model<-glm(Results~Cont, data = TD,family = binomial(link = "logit"))
modeln<-glm(Results~1, data = TD,family = binomial(link = "logit"))

anova(modeln, model, test = 'Chisq')

NagelkerkeR2(model)

#(exp(-3.4393+0.4566*0))/(1-exp(-3.4393+0.4566*0))

#good fit of the model
hoslem.test(model$y, fitted(model), g = 10)


#water testing
TD_W<-read_csv("Data/Water_Testing_Data.csv")
#TD_W$Results<-as.factor(TD_W$Results)
TD_W<-TD_W[,2:3]


model_W<-glm(Results~Cont, data = TD_W,family = binomial(link = "logit"))

model_Wn<-glm(Results~1, data = TD_W,family = binomial(link = "logit"))



TD_W$Ones <- rep(1, nrow(TD_W))

mod<-glmnet(TD_W[,c(1,3)],TD_W$Results, family= "binomial",nlambda = 1000)
plot(mod$lambda)
mod$lambda[733]

which.min(mod$lambda) 
coef(mod)[,733]





NagelkerkeR2(model_W)

anova(model_Wn, model_W, test = 'Chisq')

pchisq(40.544,1)

#also goof fit of the model
hoslem.test(model_W$y, fitted(model_W), g = 10)

(exp(-19.8+3.424*4))/(1+exp(-19.8+3.424*4))



