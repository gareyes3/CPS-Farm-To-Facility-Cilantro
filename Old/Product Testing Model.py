# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:37:33 2022

@author: gareyes3
"""

#%%
import sys
sys.path
sys.path.append('C:\\Users\Gustavo Reyes\Documents\GitHubFiles\CPS-Farm-To-Facility-Cilantro')
sys.path.append('C:\\Users\gareyes3\Documents\GitHub\CPS-Farm-To-Facility-Cilantro')
sys.path.append('C:\\Users\\reyes\\Documents\\GitHub\\CPS-Farm-To-Facility-Cilantro')

import numpy as np
import pandas as pd
import random 
import math
from numpy.random import Generator, PCG64
rng = Generator(PCG64())
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit, cuda
import pickle
#%%Functions
def random_chunk(lst, chunk_size):
    nb_chunks = int(math.ceil(len(lst)/chunk_size))
    choice = random.randrange(nb_chunks) # 0 <= choice < nb_chunks
    return lst[choice*chunk_size:(choice+1)*chunk_size]

def field_cont_percetage2(df, percent_cont, Hazard_lvl,No_Cont_Clusters):
    df2=df.copy()
    No_Cont_Clusters = 1
    #This function contaminated the tomato field randomly based on a cluter of X%. 
    Percent_Contaminated =percent_cont #Percentage of tomatoes contaminated
    Percent_D_Contaminatinated= Percent_Contaminated/100 #Percentage in decimal point
    Hazard_lvl = Hazard_lvl #CFUs in contaminated area total Cells
    
    No_Cont_PartitionUnits= int(len(df)*Percent_D_Contaminatinated)
    Field_df_1 =df.loc[(df["Location"]==1) & (df["Rej_Acc"]=="Acc")].copy()
    
    if len(Field_df_1)>0:
        Hazard_lvl_percluster= Hazard_lvl / No_Cont_Clusters #(No_Cont_PartitionUnits*No_Cont_Clusters)
        for i in range(0,No_Cont_Clusters):
            random_Chunky = np.array(random_chunk(lst = df.index, chunk_size = No_Cont_PartitionUnits)) #creating random chunk
            Contamination_Pattern = rng.multinomial(Hazard_lvl_percluster,[1/No_Cont_PartitionUnits]*No_Cont_PartitionUnits,1) #spliting cont into chunks length
            random_Chunky_s= random_Chunky[np.isin(random_Chunky,np.array(Field_df_1.index))]
            Contamination_Pattern_s = Contamination_Pattern[0][range(0,len(random_Chunky_s))]
            Field_df_1.loc[random_Chunky_s, "Oo"] = Field_df_1.loc[random_Chunky_s, "Oo"] + Contamination_Pattern_s
            #Field_df_1.loc[random_Chunky, "CFU"] = Field_df_1.loc[random_Chunky, "CFU"] + Contamination_Pattern[0] #adding contmaination
            
        df2.update(Field_df_1)
        return df2
    
#Sampling function
def Cilantro_Sampling(df , N_Grabs ,Composite_Mass ,Plant_Weight ):
    df2 = df.copy()
    grabs_weight = (Composite_Mass/N_Grabs)/454 #in lbs.     
    Oo_list=df2.loc[:,"Oo"]#list of oocyst in the field by location, vectorization
    Total_Oocyst_Grab = []
    Sample_Indeces = []
    for j in range(N_Grabs): #sampling each grab
        List_Random=Oo_list.sample(n=1) #slecting one sample from the list
        Oo_bunch = List_Random
        Oo_Sample = np.random.binomial(Oo_bunch,p=(grabs_weight/Plant_Weight))
        Index_Sample = List_Random.index[0]
        Total_Oocyst_Grab.append(Oo_Sample[0])
        Sample_Indeces.append(Index_Sample)
    
    Total_Oo_Composite = sum(Total_Oocyst_Grab)
    Pr_Detect = loaded_model.predict_proba(np.array([Total_Oo_Composite]).reshape(-1,1))[0][1]
    if np.random.uniform(0,1) < Pr_Detect:
        df2.loc[Sample_Indeces, 'PositiveSamples'] = df2.loc[Sample_Indeces, 'PositiveSamples'] + 1
    
    return df2


def F_Rejection_Rule_C (df):
    df_field_1 =df.copy()
    Postives = sum(df_field_1['PositiveSamples'] >0)
    if Postives>0:
     df_field_1.loc[:,"Rej_Acc"] = "REJ"
     df_field_1.loc[:,"Oo_BRej"] = df_field_1["Oo"]
     df_field_1.loc[:,"Oo"] = 0
    return df_field_1
    

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

loaded_model = pickle.load(open(filename, 'rb'))

loaded_model.predict_proba(np.array([10]).reshape(-1,1))[0][1]

probs_detect = []
for i in (range(200)):
    prob_detect = loaded_model.predict_proba(np.array([i]).reshape(-1,1))[0][1]
    probs_detect.append(prob_detect)

sns.lineplot(y =probs_detect, x = range(200))
plt.xlabel("OOcyst per sample")
plt.ylabel("Probability of Detection")

#%% Product Testing with fitted logistic

#Creating the Field
Field_Yield = 22_000 #lb
Plant_Weight = 1 #lb
Total_Plants = int(Field_Yield/Plant_Weight)
Total_Plants_List = range(1,Total_Plants+1)

Case_Weight = 20 #lb per case
Bunches_Weight = Case_Weight/Plant_Weight

#Sampling Characteristics Product
N_Grabs = 60
Composite_Mass = 375



'''
#Aggregative sampling. 
Oo_list=Cilantro_df.loc[:,"Oo"]
for j in range(N_Grabs): #sampling each grab
    List_Random=Oo_list.sample(n=1) #slecting one sample from the list
    Oo_bunch = List_Random
    Oo_Sample = np.random.binomial(Oo_bunch,p=(0.013766/0.25))
    Index_Sample = List_Random.index[0] 
    Pr_Detect = -1.0550* math.exp(-0.1163 * Oo_Sample) +1.0211
    if Pr_Detect<0:
        Pr_Detect= 0
    RandomUnif = random.uniform(0,1)
    if RandomUnif < Pr_Detect:
        Cilantro_df.at[Index_Sample, 'PositiveSamples'] = Cilantro_df.at[Index_Sample, 'PositiveSamples'] + 1
        print("Detected")
''' 

   



#Water Characteristics
Water_Irrigation_In = 12 #Inches of water per harvest season
Total_L_Season = 40.46*40.46*(0.0254*Water_Irrigation_In)*1000 # one acre 40.46m2 * 0.348 m of water * 10000 to convert tot m3
Days_per_season = 45 #days
L_water_day = Total_L_Season/Days_per_season

#Water contamintion
OO_per_L = 0.6
Initial_Levels_Bulk = int(Total_L_Season*OO_per_L)
Initial_Levels_Bulk_Day = L_water_day*OO_per_L


#Creation of data frame
Cilantro_df=pd.DataFrame({"Plant_ID": Total_Plants_List,
                       "Weight": Plant_Weight,
                       "Case_PH": 0,
                       "Oo": 0,
                       "Oo_BRej":"",
                       "Location": 1,
                       'PositiveSamples':0,
                       "Rej_Acc" :"Acc"
                  })

#contamination
Cilantro_df= field_cont_percetage2(df = Cilantro_df, 
                                   percent_cont = 100, 
                                   Hazard_lvl = Initial_Levels_Bulk_Day,
                                   No_Cont_Clusters = 1)
   

Cilantro_df= Cilantro_Sampling(df  = Cilantro_df, 
                  N_Grabs = N_Grabs ,
                  Composite_Mass=Composite_Mass,
                  Plant_Weight = Plant_Weight ) 

Cilantro_df=F_Rejection_Rule_C (df = Cilantro_df)

len(Cilantro_df[Cilantro_df["Rej_Acc"] == "REJ"])
    
    
#%%
#Analysis Function

#Water Characteristics
Water_Irrigation_In = 12 #Inches of water per harvest season
Total_L_Season = 40.46*40.46*(0.0254*Water_Irrigation_In)*1000 # one acre 40.46m2 * 0.348 m of water * 10000 to convert tot m3
Days_per_season = 45 #days
L_water_day = Total_L_Season/Days_per_season

#Water contamintion
OO_per_L = 20
Initial_Levels_Bulk = int(Total_L_Season*OO_per_L)
Initial_Levels_Bulk_Day = L_water_day*OO_per_L

Cilantro_df=pd.DataFrame({"Plant_ID": Total_Plants_List,
                       "Weight": Plant_Weight,
                       "Case_PH": 0,
                       "Oo": 0,
                       "Oo_BRej":"",
                       "Location": 1,
                       'PositiveSamples':0,
                       "Rej_Acc" :"Acc"
                  })
        


def Samling_Var(Cilantro_df, Hazard_lvl, percent_cont, N_Grabs, Composite_Mass, Plant_Weight,Field_Iters = 100, Sampling_Iters = 100):  
    P_Reject = []
    for j in range(Field_Iters):
        print(j)
        #Creation of data frame
        #contamination
        Cilantro_df_F= field_cont_percetage2(df = Cilantro_df, 
                                           percent_cont = percent_cont, 
                                           Hazard_lvl = Hazard_lvl,
                                           No_Cont_Clusters = 1)
        L_Acc_Reject = []
        for i in range(Sampling_Iters):#Sampling Uncertainty Iterations.

            Cilantro_df_I= Cilantro_Sampling(df  = Cilantro_df_F, 
                              N_Grabs = N_Grabs,
                              Composite_Mass=Composite_Mass,
                              Plant_Weight = Plant_Weight ) 
            
            Cilantro_df_I=F_Rejection_Rule_C (df = Cilantro_df_I)
            
            if len(Cilantro_df_I[Cilantro_df_I["Rej_Acc"] == "REJ"])>0:
                L_Acc_Reject.append(1)
            else:
                L_Acc_Reject.append(0) 
        P_Reject.append(np.mean(L_Acc_Reject)) 
    return P_Reject




Df_Outs = pd.DataFrame(columns=["PDetect", "Conts", "Grabs"])
Total_grabs = [15,30,60]

for k in Total_grabs:
    Cont_Levels = list(np.linspace(1,1_000_000, 20)) 
    
    Probs_L = []
    for i in Cont_Levels:
        Probs = Samling_Var(Cilantro_df =Cilantro_df , 
                    Hazard_lvl =i, 
                    percent_cont = 100,
                    N_Grabs = k , 
                    Composite_Mass = k*25 , 
                    Plant_Weight = 1,
                    Field_Iters = 10, 
                    Sampling_Iters = 10)
        Probs_L.append(Probs)
        
    Df_Outs2 = pd.DataFrame({
        "PDetect":np.concatenate(Probs_L).flat,
        "Conts": np.repeat(Cont_Levels,10),
        "Grabs" : k
        })
    
    Df_Outs= pd.concat([Df_Outs,Df_Outs2])
    

Df_Outs.reset_index(inplace=True, drop = True) 


sns.lineplot(data =Df_Outs, x = "Conts", y = "PDetect", hue = "Grabs" )


    