# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 16:01:27 2022

@author: Gustavo Reyes
"""
#%%
import sys
sys.path
sys.path.append('C:\\Users\Gustavo Reyes\Documents\GitHubFiles\CPS-Farm-To-Facility-Cilantro')
sys.path.append('C:\\Users\gareyes3\Documents\GitHub\CPS-Farm-To-Facility-Cilantro')
sys.path.append('C:\\Users\\reyes\\Documents\\GitHub\\CPS-Farm-To-Facility-Cilantro')

import numpy as np
import pandas as pd
from numpy.random import Generator, PCG64
rng = Generator(PCG64())
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import statsmodels.api as smf
import math
#%%

###############################################################################

#Product Testing Model Logistic nPCR FDA, not used old method. 
R1=np.concatenate([np.ones(0), np.zeros(39)])
R2=np.concatenate([np.ones(12), np.zeros(28)])
R3=np.concatenate([np.ones(29), np.zeros(11)])
R4=np.concatenate([np.ones(40), np.zeros(0)])


T1=np.repeat(0,39)
T2=np.repeat(5,40)
T3=np.repeat(10,40)
T4=np.repeat(200,40)

Df_PD=pd.DataFrame({
        "Cont": np.concatenate([T1,T2,T3,T4]),
        "Results": np.concatenate([R1,R2,R3,R4]),
        })

Df_PD.to_csv("C:\\Users\\Gustavo Reyes\\Documents\\GitHubFiles\\CPS-Farm-To-Facility-Cilantro\\Data_Cilantro_Outputs\\Product_Testing_Data.csv")


sns.scatterplot(data =Df_PD, x ="Cont", y=  "Results" )

from sklearn import linear_model
logr = linear_model.LogisticRegression()
logr.fit(np.array(Df_PD["Cont"]).reshape(-1,1),np.array(Df_PD["Results"]))

logr.score(np.array(Df_PD["Cont"]).reshape(-1,1),np.array(Df_PD["Results"]))

logr.predict_proba(np.array([0]).reshape(-1,1))[0][1]
logr.intercept_
logr.coef_


def deviance(X, y, model):
    return 2*metrics.log_loss(y, model.predict_proba(X), normalize=False)

def loglike(X, y, model):
    return metrics.log_loss(y, model.predict_proba(X), normalize=False)

#chisqauered for the null model vs the full model
deviance(np.array(Df_PD["Cont"]).reshape(-1,1),np.array(Df_PD["Results"]), logr)


from sklearn.dummy import DummyClassifier
from sklearn import metrics
import scipy
logr_D = DummyClassifier()
logr_D.fit(np.array(Df_PD["Cont"]).reshape(-1,1),np.array(Df_PD["Results"]))

deviance(np.array(Df_PD["Cont"]).reshape(-1,1),np.array(Df_PD["Results"]), logr_D)

#chi sqaured.
scipy.stats.chi2.sf((220.36419628552224-99.34764524320522), 1)
#2.2328359100593915e-10


# save the model to disk
filename = 'C:\\Users\gareyes3\Documents\GitHub\CPS-Farm-To-Facility-Cilantro\logistic_Prod_Test_nPCR_FDA.sav'
pickle.dump(logr, open(filename, 'wb'))

np.arange(0, 200, 0.01)
probs_detect = []
for i in (np.arange(0, 200, 0.01)):
    prob_detect = logr.predict_proba(np.array([i]).reshape(-1,1))[0][1]
    probs_detect.append(prob_detect)

sns.scatterplot(data =Df_PD ,x ="Cont", y=  "Results" )
sns.lineplot(y =probs_detect, x = np.arange(0, 200, 0.01))
plt.xlabel("OOcyst per 25g sample")
plt.ylabel("Probability of Detection")
plt.title("Product Testing nPCR Method")


##############################################################################

#Product Testing Model Logistic qPCR FDA USED in Model**** From Murphy 2018, new method, Lab A and B Data combined. 

R1_qPCR=np.concatenate([np.ones(0), np.zeros(80)])
R2_qPCR=np.concatenate([np.ones(25), np.zeros(55)])
R3_qPCR=np.concatenate([np.ones(64), np.zeros(16)])
R4_qPCR=np.concatenate([np.ones(80), np.zeros(0)])


T1_qPCR=np.repeat(0,80)
T2_qPCR=np.repeat(5,80)
T3_qPCR=np.repeat(10,80)
T4_qPCR=np.repeat(200,80)

Df_PD_qPCR=pd.DataFrame({
        "Cont": np.concatenate([T1_qPCR,T2_qPCR,T3_qPCR,T4_qPCR]),
        "Results": np.concatenate([R1_qPCR,R2_qPCR,R3_qPCR,R4_qPCR]),
        })

#Fitting the logistic regression model
from sklearn import linear_model
#Logistic Regression
logr_qPCR = linear_model.LogisticRegression()
logr_qPCR.fit(np.array(Df_PD_qPCR["Cont"]).reshape(-1,1),np.array(Df_PD_qPCR["Results"]))
logr_qPCR.score(np.array(Df_PD_qPCR["Cont"]).reshape(-1,1),np.array(Df_PD_qPCR["Results"]))

logr_qPCR.intercept_
logr_qPCR.coef_

from sklearn import metrics

def deviance(X, y, model):
    return 2*metrics.log_loss(y, model.predict_proba(X), normalize=False)

def loglike(X, y, model):
    return metrics.log_loss(y, model.predict_proba(X), normalize=False)

#chisqauered for the null model vs the full model
deviance(np.array(Df_PD_qPCR["Cont"]).reshape(-1,1),np.array(Df_PD_qPCR["Results"]), logr_qPCR)

from sklearn.dummy import DummyClassifier
import scipy
logr_D_qPRC = DummyClassifier()
logr_D_qPRC.fit(np.array(Df_PD_qPCR["Cont"]).reshape(-1,1),np.array(Df_PD_qPCR["Results"]))

deviance(np.array(Df_PD_qPCR["Cont"]).reshape(-1,1),np.array(Df_PD_qPCR["Results"]), logr_D_qPRC)

#chi sqaured.
scipy.stats.chi2.sf(( 442.6011609459082-185.00983117735439), 1)
#5.7485453799292104e-58 goodness of fit satisfied. 

# save the model to disk
filename = 'C://Users/gareyes3/Documents/GitHub/CPS-Farm-To-Facility-Cilantro/logistic_Prod_Test_qPCR_FDA.sav'
pickle.dump(logr_qPCR, open(filename, 'wb'))


#Trying other classifiers, not relevant for manuscript
'''
from sklearn.naive_bayes import GaussianNB
nbayes_qPCR = GaussianNB()
nbayes_qPCR.fit(np.array(Df_PD_qPCR["Cont"]).reshape(-1,1),np.array(Df_PD_qPCR["Results"]))
nbayes_qPCR.score(np.array(Df_PD_qPCR["Cont"]).reshape(-1,1),np.array(Df_PD_qPCR["Results"]))

from sklearn.ensemble import RandomForestClassifier
clf_qPCR = RandomForestClassifier()
clf_qPCR.fit(np.array(Df_PD_qPCR["Cont"]).reshape(-1,1),np.array(Df_PD_qPCR["Results"]))
clf_qPCR.score(np.array(Df_PD_qPCR["Cont"]).reshape(-1,1),np.array(Df_PD_qPCR["Results"]))
clf_qPCR.predict_proba(np.array([7]).reshape(-1,1))[0][1]

from sklearn import tree
tree_qPCR = tree.DecisionTreeClassifier()
tree_qPCR.fit(np.array(Df_PD_qPCR["Cont"]).reshape(-1,1),np.array(Df_PD_qPCR["Results"]))
tree_qPCR.score(np.array(Df_PD_qPCR["Cont"]).reshape(-1,1),np.array(Df_PD_qPCR["Results"]))
tree_qPCR.predict_proba(np.array([7]).reshape(-1,1))[0][1]

from sklearn import svm
svm_qPCR = svm.SVC(probability=True)
svm_qPCR.fit(np.array(Df_PD_qPCR["Cont"]).reshape(-1,1),np.array(Df_PD_qPCR["Results"]))
svm_qPCR.score(np.array(Df_PD_qPCR["Cont"]).reshape(-1,1),np.array(Df_PD_qPCR["Results"]))
svm_qPCR.predict_proba(np.array([7]).reshape(-1,1))[0][1]

logr_qPCR.predict_proba(np.array([10]).reshape(-1,1))[0][1]
'''
probs_detect = []
for i in (range(200)):
    prob_detect = logr_qPCR.predict_proba(np.array([i]).reshape(-1,1))[0][1]
    probs_detect.append(prob_detect)

sns.scatterplot(data =Df_PD_qPCR, x ="Cont", y=  "Results" ,x_jitter=75)
sns.lineplot(y =probs_detect, x = range(200))
plt.xlabel("Oocyst per 25g sample")
plt.ylabel("Probability of Detection")
plt.title("Product Testing qPCR Method")

Df_PD_qPCR.to_csv("C:\\Users\\Gustavo Reyes\Documents\GitHubFiles\\CPS-Farm-To-Facility-Cilantro\\Data\\qPCR_Fit_Product_Testing.csv")
Df_probs_qPCR = pd.DataFrame({"Prob Detect":probs_detect,
                             "Cont" : range(200)})

Df_probs_qPCR.to_csv("C:\\Users\\Gustavo Reyes\Documents\GitHubFiles\\CPS-Farm-To-Facility-Cilantro\\Data\\qPCR_Fit_Product_Testing_Probs.csv")


probs_detect = []
for i in (range(200)):
    prob_detect =clf_qPCR.predict_proba(np.array([i]).reshape(-1,1))[0][1]
    probs_detect.append(prob_detect)

sns.scatterplot(data =Df_PD_qPCR, x ="Cont", y=  "Results" )
sns.lineplot(y =probs_detect, x = range(200))
plt.xlabel("OOcyst per 25g sample")
plt.ylabel("Probability of Detection")
plt.title("Product Testing qPCR Method")

#Product Testing Model Logistic QPRCR 2021 Paper 2. Costa
R1_qPCR_2=np.concatenate([np.ones(0), np.zeros(5)])
R2_qPCR_2=np.concatenate([np.ones(1), np.zeros(5)])
R3_qPCR_2=np.concatenate([np.ones(3), np.zeros(1)])
R4_qPCR_2=np.concatenate([np.ones(5), np.zeros(0)])


T1_qPCR_2=np.repeat(0,5)
T2_qPCR_2=np.repeat(5,6)
T3_qPCR_2=np.repeat(10,4)
T4_qPCR_2=np.repeat(200,5)

Df_PD_qPCR_2=pd.DataFrame({
        "Cont": np.concatenate([T1_qPCR,T2_qPCR,T3_qPCR,T4_qPCR]),
        "Results": np.concatenate([R1_qPCR,R2_qPCR,R3_qPCR,R4_qPCR]),
        })

#from statsmodels.discrete.discrete_model import Probit
from sklearn import linear_model
logr_qPCR_2 = linear_model.LogisticRegression()
logr_qPCR_2.fit(np.array(Df_PD_qPCR_2["Cont"]).reshape(-1,1),np.array(Df_PD_qPCR_2["Results"]))
#Provit_Model= Probit(Df_PD_qPCR_2["Results"], Df_PD_qPCR_2["Cont"]).fit()
#print(Provit_Model.summary())
#print(Provit_Model.predict(11.19))
# save the model to disk
filename = 'C:\\Users\gareyes3\Documents\GitHub\CPS-Farm-To-Facility-Cilantro\logistic_Prod_Test_qPCR_2_Costa.sav'
pickle.dump(logr_qPCR, open(filename, 'wb'))


probs_detect = []
for i in (range(200)):
   # prob_detect = logr_qPCR_2.predict_proba(np.array([i]).reshape(-1,1))[0][1]
    prob_detect = logr_qPCR_2.predict(i)[0]
    probs_detect.append(prob_detect)

sns.scatterplot(data =Df_PD_qPCR, x ="Cont", y=  "Results" )
sns.lineplot(y =probs_detect, x = range(200))
plt.xlabel("OOcyst per 25g sample")
plt.ylabel("Probability of Detection")
plt.title("Product Testing qPCR Method Paper 2")

#%%
#Agricultura Water Logisitc

R1_AW=np.concatenate([np.ones(0), np.zeros(12)])
R2_AW=np.concatenate([np.ones(8), np.zeros(4)])
R3_AW=np.concatenate([np.ones(3), np.zeros(0)])
R4_AW=np.concatenate([np.ones(6), np.zeros(0)])
R5_AW=np.concatenate([np.ones(3), np.zeros(0)])
R6_AW=np.concatenate([np.ones(6), np.zeros(0)])


T1_AW=np.repeat(0,12)
T2_AW=np.repeat(6,12)
T3_AW=np.repeat(12,3)
T4_AW=np.repeat(25,6)
T5_AW=np.repeat(100,3)
T6_AW=np.repeat(200,6)

Df_PD_AW=pd.DataFrame({
        "Cont": np.concatenate([T1_AW,T2_AW,T3_AW,T4_AW,T5_AW,T6_AW]),
        "Results": np.concatenate([R1_AW,R2_AW,R3_AW,R4_AW,R5_AW,R6_AW]),
        })

Df_PD_AW.to_csv("C:\\Users\\gareyes3\\Documents\\GitHub\\CPS-Farm-To-Facility-Cilantro\\Data\\Water_Testing_Data.csv")
Df_PD_AW['intercept'] = 1.0

from sklearn import linear_model
from sklearn import metrics
import scipy
logr_AW = linear_model.LogisticRegression()
logr_AW.fit(np.array(Df_PD_AW["Cont"]).reshape(-1,1),np.array(Df_PD_AW["Results"]))


logr_AW.score(np.array(Df_PD_AW["Cont"]).reshape(-1,1),np.array(Df_PD_AW["Results"]))

logr_AW.intercept_
logr_AW.coef_

def deviance(X, y, model):
    return 2*metrics.log_loss(y, model.predict_proba(X), normalize=False)

def loglike(X, y, model):
    return metrics.log_loss(y, model.predict_proba(X), normalize=False)

#chisqauered for the null model vs the full model
deviance(np.array(Df_PD_AW["Cont"]).reshape(-1,1), np.array(Df_PD_AW["Results"]), logr_AW)


from sklearn.dummy import DummyClassifier
logr_AW_D = DummyClassifier()
logr_AW_D.fit(np.array(Df_PD_AW["Cont"]).reshape(-1,1),np.array(Df_PD_AW["Results"]))

deviance(np.array(Df_PD_AW["Cont"]).reshape(-1,1), np.array(Df_PD_AW["Results"]), logr_AW_D)

#chi sqaured.
scipy.stats.chi2.sf((55.82038884701287-15.568886670709356), 1)
#2.2328359100593915e-10


# importing libraries
import statsmodels.api as sm
import pandas as pd 
  
# defining the dependent and independent variables
Xtrain = Df_PD_AW[['intercept',"Cont"]]
ytrain = Df_PD_AW[['Results']]
   
# building the model and fitting the data
log_reg = sm.Logit(ytrain, Xtrain).fit()

log_reg.predict(6)

print(log_reg.summary())


(math.exp(-4.4879239+0.85580838*0))/(1+(math.exp(-4.4879239+0.85580838*0)))

0.02577*32
1-(1-0.02577)**16


import statsmodels.api as sm
log_reg = sm.Logit(Df_PD_AW["Results"],Df_PD_AW["Cont"]).fit()
print(log_reg.summary())


probs_detect = []
for i in (range(200)):
    prob_detect = logr_AW.predict_proba(np.array([i]).reshape(-1,1))[0][1]
    probs_detect.append(prob_detect)
    

sns.scatterplot(data =Df_PD_AW, x ="Cont", y=  "Results" )
sns.lineplot(y =probs_detect, x = range(200))
plt.xlabel("OOcyst per 10L sample")
plt.ylabel("Probability of Detection")
plt.title("Agricultural Water Fit")

Df_PD_AW.to_csv("C:\\Users\\gareyes3\\Documents\\GitHub\\CPS-Farm-To-Facility-Cilantro\\Data_Cilantro_Outputs\\qPCR_Fit_Water_Testing.csv")
Df_probs_logr_AW = pd.DataFrame({"Prob Detect":probs_detect,
                             "Cont" : range(200)})

Df_probs_logr_AW.to_csv("C:\\Users\\gareyes3\\Documents\\GitHub\\CPS-Farm-To-Facility-Cilantro\\Data_Cilantro_Outputs\\qPCR_Fit_Water_Testing_Probs.csv")



# save the model to disk
filename = 'C:\\Users\gareyes3\Documents\GitHub\CPS-Farm-To-Facility-Cilantro\logistic_AW_Testing_qPCR.sav'
pickle.dump(logr_AW , open(filename, 'wb'))
