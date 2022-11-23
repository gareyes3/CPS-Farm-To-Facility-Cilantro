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
#%%

#Product Testing Model Logistic
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

sns.scatterplot(data =Df_PD, x ="Cont", y=  "Results" )

from sklearn import linear_model
logr = linear_model.LogisticRegression()
logr.fit(np.array(Df_PD["Cont"]).reshape(-1,1),np.array(Df_PD["Results"]))

# save the model to disk
filename = 'C:\\Users\gareyes3\Documents\GitHub\CPS-Farm-To-Facility-Cilantro\logistic_Prod_Test.sav'
pickle.dump(logr, open(filename, 'wb'))

probs_detect = []
for i in (range(200)):
    prob_detect = logr.predict_proba(np.array([i]).reshape(-1,1))[0][1]
    probs_detect.append(prob_detect)

sns.lineplot(y =probs_detect, x = range(200))
plt.xlabel("OOcyst per sample")
plt.ylabel("Probability of Detection")

#%%
#Other qPCR

R1=np.concatenate([np.ones(0), np.zeros(12)])
R2=np.concatenate([np.ones(8), np.zeros(4)])
R3=np.concatenate([np.ones(3), np.zeros(0)])
R4=np.concatenate([np.ones(6), np.zeros(0)])
R5=np.concatenate([np.ones(3), np.zeros(0)])
R6=np.concatenate([np.ones(6), np.zeros(0)])


T1=np.repeat(0,12)
T2=np.repeat(6,8)
T3=np.repeat(12,3)
T4=np.repeat(25,6)
T5=np.repeat(100,3)
T6=np.repeat(200,6)

Df_PD=pd.DataFrame({
        "Cont": np.concatenate([T1,T2,T3,T4,T5,T6]),
        "Results": np.concatenate([R1,R2,R3,R4,R5,R6]),
        })

sns.scatterplot(data =Df_PD, x ="Cont", y=  "Results" )

from sklearn import linear_model
logr = linear_model.LogisticRegression()
logr.fit(np.array(Df_PD["Cont"]).reshape(-1,1),np.array(Df_PD["Results"]))

probs_detect = []
for i in (range(200)):
    prob_detect = logr.predict_proba(np.array([i]).reshape(-1,1))[0][1]
    probs_detect.append(prob_detect)

sns.lineplot(y =probs_detect, x = range(200))
plt.xlabel("OOcyst per sample")
plt.ylabel("Probability of Detection")
