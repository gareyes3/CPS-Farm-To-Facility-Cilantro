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
modeln<-glm(Results~1, data = TD,family = binomial(link = "logit"))

anova(modeln, model, test = 'Chisq')

NagelkerkeR2(model)

220.4 -99.35

(exp(-3.4393+0.4566*0))/(1-exp(-3.4393+0.4566*0))

predict(model,data_predict )

hoslem.test(model$y, fitted(model), g = 10)


#water testing
TD_W<-read_csv("Water_Testing_Data.csv")
TD_W$Results<-as.factor(TD_W$Results)
TD_W<-TD_W[,2:3]

model_W<-glm(Results~Cont, data = TD_W,family = binomial(link = "logit"))
model_Wn<-glm(Results~1, data = TD_W,family = binomial(link = "logit"))

NagelkerkeR2(model_W)

anova(model_Wn, model_W, test = 'Chisq')
hoslem.test(model_W$y, fitted(model_W), g = 10)

(exp(-19.8+3.424*0))/(1-exp(-19.8+3.424  *0))



