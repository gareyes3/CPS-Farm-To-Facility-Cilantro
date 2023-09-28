# -*- coding: utf-8 -*-
"""
This document creates the analysis to test number of grabs
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
    No_Cont_PartitionUnits= int(len(df)*(percent_cont/100))
    Field_df_1 =df.loc[(df["Location"]==1) & (df["Rej_Acc"]=="Acc")].copy()
    if len(Field_df_1)>0:
        Hazard_lvl_percluster= Hazard_lvl / No_Cont_Clusters #(No_Cont_PartitionUnits*No_Cont_Clusters)
        for i in range(0,No_Cont_Clusters):
            random_Chunky = np.array(random_chunk(lst = df.index, chunk_size = No_Cont_PartitionUnits)) #creating random chunk
            Contamination_Pattern = rng.multinomial(Hazard_lvl_percluster,[1/No_Cont_PartitionUnits]*No_Cont_PartitionUnits,1) #spliting cont into chunks length
            random_Chunky_s= random_Chunky[np.isin(random_Chunky,np.array(Field_df_1.index))]
            Contamination_Pattern_s = Contamination_Pattern[0][range(0,len(random_Chunky_s))]
            Field_df_1.loc[random_Chunky_s, "Oo"] = Field_df_1.loc[random_Chunky_s, "Oo"] + Contamination_Pattern_s
            
        df2.update(Field_df_1)
        return df2
    
#Sampling function
def Cilantro_Sampling(df , N_Grabs ,Composite_Mass ,Plant_Weight, loaded_model ):
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

def Cilantro_Sampling_25g(df,Sample_Weight,N_25g_Samples,N_Grabs_Sample ,Plant_Weight, loaded_model ):
    df2 = df.copy()
    grabs_weight = (Sample_Weight/N_Grabs_Sample)/454 #in lbs.     
    Oo_list=df2.loc[:,"Oo"]#list of oocyst in the field by location, vectorization
    for i in range(N_25g_Samples):
        Total_Oocyst_Grab = []
        Sample_Indeces = []
        for j in range(N_Grabs_Sample): #sampling each grab
            List_Random=Oo_list.sample(n=1) #slecting one sample from the list
            Oo_bunch = List_Random
            Oo_Sample = np.random.binomial(Oo_bunch,p=(grabs_weight/Plant_Weight))
            Total_Oocyst_Grab.append(Oo_Sample[0])
            Sample_Indeces.append(List_Random.index[0])
        Total_Oo_Composite = sum(Total_Oocyst_Grab)
        if Total_Oo_Composite>=1:
            Pr_Detect = loaded_model.predict_proba(np.array([Total_Oo_Composite]).reshape(-1,1))[0][1] #from logistic
        else:
            Pr_Detect = 0
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
#Loading the saved logistic regression model

#filename = 'C:\\Users\gareyes3\Documents\GitHub\CPS-Farm-To-Facility-Cilantro\logistic_Prod_Test.sav'
filename_nPCR = 'C:\\Users\Gustavo Reyes\Documents\GitHubFiles\CPS-Farm-To-Facility-Cilantro\logistic_Prod_Test_nPCR_FDA.sav'
nPCR_Model = pickle.load(open(filename_nPCR, 'rb'))

filename_qPCR = 'C:\\Users\Gustavo Reyes\Documents\GitHubFiles\CPS-Farm-To-Facility-Cilantro\logistic_Prod_Test_qPCR_FDA.sav'
qPCR_Model = pickle.load(open(filename_qPCR, 'rb'))


#%%
#Product Testing Module

#Function to simulate variation
#default 100 sampling plans 
#default 100 fields simulated
def Samling_Var(Cilantro_df, Hazard_lvl, percent_cont, Sample_Weight,N_25g_Samples,N_Grabs_Sample, Plant_Weight,loaded_model,Field_Iters = 100, Sampling_Iters = 100):  
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
            '''
            Cilantro_df_I= Cilantro_Sampling(df  = Cilantro_df_F, 
                              N_Grabs = N_Grabs,
                              Composite_Mass=Composite_Mass,
                              Plant_Weight = Plant_Weight,
                              loaded_model = loaded_model) 
            '''
            Cilantro_df_I= Cilantro_Sampling_25g(df = Cilantro_df_F,
                                  Sample_Weight =Sample_Weight,
                                  N_25g_Samples=N_25g_Samples,
                                  N_Grabs_Sample=N_Grabs_Sample,
                                  Plant_Weight=Plant_Weight, 
                                  loaded_model=loaded_model)
            
            Cilantro_df_I=F_Rejection_Rule_C (df = Cilantro_df_I)
            
            if len(Cilantro_df_I[Cilantro_df_I["Rej_Acc"] == "REJ"])>0:
                L_Acc_Reject.append(1)
            else:
                L_Acc_Reject.append(0) 
        P_Reject.append(np.mean(L_Acc_Reject)) 
    return P_Reject


#Field and Model Characyeristics: 
    
#Creating the Field
Field_Yield = 22_000 #lb
Plant_Weight = 1 #lb
Total_Plants = int(Field_Yield/Plant_Weight)
Total_Plants_List = range(1,Total_Plants+1)

Case_Weight = 20 #lb per case
Bunches_Weight = Case_Weight/Plant_Weight


#Water Characteristics, not necesarily used. 
Water_Irrigation_In = 12 #Inches of water per harvest season
Total_L_Season = 63.6*63.6*(0.0254*Water_Irrigation_In)*1000 # one acre 40.46m2 * 0.348 m of water * 10000 to convert tot m3
Days_per_season = 45 #days
L_water_day = Total_L_Season/Days_per_season

#Water contamintion to calculate min and max. 
OO_per_L = 0.6
OO_per_H = 20

Initial_Levels_Bulk_Low = int(Total_L_Season*OO_per_L)
Initial_Levels_Bulk_Day_Low = L_water_day*OO_per_L
#299376, Oo per season
#6652.8, Oo per day
Initial_Levels_Bulk_High = int(Total_L_Season*OO_per_H)
Initial_Levels_Bulk_Day_High = L_water_day*OO_per_H

#9979222, Oo per season
#221760, Oo per day

#So form this we can define that best case scenario 6,652 Oo and worse case scenario 9,979,222
# we can define a range between 0 and 10_000_000 Oo iun the field. 


#%%
#Analysis: 
#function that conducts grabs variability over a vector of contaminations. 
def Iter_Cont_Levels(Cont_Levels, Cilantro_df, percent_cont, Sample_Weight,N_25g_Samples,N_Grabs_Sample, Plant_Weight,loaded_model, Field_Iters, Sampling_Iters   ):
    Df_Outs = pd.DataFrame(columns=["PDetect", "Conts", "N25gsamples"]) #creating empty collection dataframe
    Probs_L = []
    for i in Cont_Levels:
        Probs = Samling_Var(Cilantro_df =Cilantro_df , 
                    Hazard_lvl =i, 
                    percent_cont = percent_cont,
                    Sample_Weight=Sample_Weight,
                    N_25g_Samples =N_25g_Samples,
                    N_Grabs_Sample =N_Grabs_Sample,
                    Plant_Weight = Plant_Weight ,
                    loaded_model= loaded_model,
                    Field_Iters = Field_Iters, 
                    Sampling_Iters = Sampling_Iters)
        Probs_L.append(Probs)
        
    Df_Outs2 = pd.DataFrame({
        "PDetect":np.concatenate(Probs_L).flat,
        "Conts": np.repeat(Cont_Levels,Field_Iters),
        "N25gsamples" :N_25g_Samples
        })
    
    Df_Outs= pd.concat([Df_Outs,Df_Outs2])
    return Df_Outs    

    
#Creating the dataframe
Cilantro_df=pd.DataFrame({"Plant_ID": Total_Plants_List,
                       "Weight": Plant_Weight,
                       "Case_PH": 0,
                       "Oo": 0,
                       "Oo_BRej":"",
                       "Location": 1,
                       'PositiveSamples':0,
                       "Rej_Acc" :"Acc"
                  })

#%%
#Grabs, each grab = 25g. 

#Cont_Levels = list(np.linspace(1,10_000_000, 100)) 
#Cont_Levels_log10= list(np.arange(3,7.5, 0.1)) 
#Cont_Levels_log10_Num =[10**x for x in Cont_Levels_log10 ]



Cont_Levels_log10_Num=[22000*454*x for x in list(np.arange(0,1.7, 0.01))]
#Cont_Levels_log10_Num=[22000*454*x for x in list(np.arange(0,0.5, 0.01))]

#%%
Outs_32g = Iter_Cont_Levels(Cont_Levels = Cont_Levels_log10_Num , 
           Cilantro_df = Cilantro_df, 
           percent_cont =100, 
           Sample_Weight =25,
           N_25g_Samples=1,
           N_Grabs_Sample =1,
           Plant_Weight = 1,
           loaded_model =  qPCR_Model,
           Field_Iters =1, 
           Sampling_Iters =1000)

Outs_32g.to_csv("C:\\Users\\gareyes3\\Documents\\GitHub\\CPS-Farm-To-Facility-Cilantro\\Data_Cilantro_Outputs\\Product_Testing_Meds.csv")


#%%
#Running Analysys for 25g grabs

Outs_32g = Iter_Cont_Levels(Cont_Levels = Cont_Levels_log10_Num , 
           Cilantro_df = Cilantro_df, 
           percent_cont =100, 
           Sample_Weight =25,
           N_25g_Samples=32,
           N_Grabs_Sample =1,
           Plant_Weight = 1,
           loaded_model = qPCR_Model,
           Field_Iters =100, 
           Sampling_Iters =100)

Outs_16g = Iter_Cont_Levels(Cont_Levels = Cont_Levels_log10_Num , 
           Cilantro_df = Cilantro_df, 
           percent_cont =100, 
           Sample_Weight =25,
           N_25g_Samples=16,
           N_Grabs_Sample =1,
           Plant_Weight = 1,
           loaded_model = qPCR_Model,
           Field_Iters =100, 
           Sampling_Iters =100)

Outs_8g = Iter_Cont_Levels(Cont_Levels =Cont_Levels_log10_Num, 
           Cilantro_df = Cilantro_df, 
           percent_cont =100, 
           Sample_Weight =25,
           N_25g_Samples=8,
           N_Grabs_Sample =1,
           Plant_Weight = 1,
           loaded_model = qPCR_Model,
           Field_Iters =100, 
           Sampling_Iters =100)


Outs_4g = Iter_Cont_Levels(Cont_Levels = Cont_Levels_log10_Num, 
           Cilantro_df = Cilantro_df, 
           percent_cont =100, 
           Sample_Weight =25,
           N_25g_Samples=4,
           N_Grabs_Sample =1,
           Plant_Weight = 1,
           loaded_model = qPCR_Model,
           Field_Iters =100, 
           Sampling_Iters =100)

Outs_2g = Iter_Cont_Levels(Cont_Levels = Cont_Levels_log10_Num, 
           Cilantro_df = Cilantro_df, 
           percent_cont =100, 
           Sample_Weight =25,
           N_25g_Samples=2,
           N_Grabs_Sample =1,
           Plant_Weight = 1,
           loaded_model = qPCR_Model,
           Field_Iters =100, 
           Sampling_Iters =100)

Outs_1g = Iter_Cont_Levels(Cont_Levels = Cont_Levels_log10_Num, 
           Cilantro_df = Cilantro_df, 
           percent_cont =100, 
           Sample_Weight =25,
           N_25g_Samples=1,
           N_Grabs_Sample =1,
           Plant_Weight = 1,
           loaded_model = qPCR_Model,
           Field_Iters =100, 
           Sampling_Iters =100)




Grabs_Combined =pd.concat([Outs_1g,Outs_2g,Outs_4g,Outs_8g,Outs_16g,Outs_32g])
Grabs_Combined.reset_index(drop= True, inplace= True)

Grabs_Combined.to_csv("C:\\Users\\Gustavo Reyes\\Documents\\GitHubFiles\\CPS-Farm-To-Facility-Cilantro\\Data_Cilantro_Outputs\\Product_Testing_Analysis_R6.csv")


#%%
#Analysis: 
#function that conducts grabs variability over a vector of contaminations. 
def Iter_Grabs(Cont_Level, Cilantro_df, percent_cont, Sample_Weight,N_25g_Samples,N_Grabs_Sample, Plant_Weight,loaded_model, Field_Iters, Sampling_Iters   ):
    Df_Outs = pd.DataFrame(columns=["PDetect", "Conts", "N25gsamples"]) #creating empty collection dataframe
    Probs_L = []
    for i in N_25g_Samples:
        Probs = Samling_Var(Cilantro_df =Cilantro_df , 
                    Hazard_lvl =Cont_Level, 
                    percent_cont = percent_cont,
                    Sample_Weight=Sample_Weight,
                    N_25g_Samples =i,
                    N_Grabs_Sample =N_Grabs_Sample,
                    Plant_Weight = Plant_Weight ,
                    loaded_model= loaded_model,
                    Field_Iters = Field_Iters, 
                    Sampling_Iters = Sampling_Iters)
        Probs_L.append(Probs)
        
    Df_Outs2 = pd.DataFrame({
        "PDetect":np.concatenate(Probs_L).flat,
        "Conts":Cont_Level ,
        "N25gsamples" :np.repeat(N_25g_Samples,Field_Iters)
        })
    
    Df_Outs= pd.concat([Df_Outs,Df_Outs2])
    return Df_Outs  


Cilantro_df=pd.DataFrame({"Plant_ID": Total_Plants_List,
                       "Weight": Plant_Weight,
                       "Case_PH": 0,
                       "Oo": 0,
                       "Oo_BRej":"",
                       "Location": 1,
                       'PositiveSamples':0,
                       "Rej_Acc" :"Acc"
                  })

N_Samples =[1,5,10,15,30,60]
Cont_Levels = [1000,10_000,100_000,1_000_000,5_000_000,10_000_000]



Outs_1 = Iter_Grabs(Cont_Level= Cont_Levels[0] , 
           Cilantro_df = Cilantro_df, 
           percent_cont =100, 
           Sample_Weight =25,
           N_25g_Samples=N_Samples,
           N_Grabs_Sample =1,
           Plant_Weight = 1,
           loaded_model = qPCR_Model,
           Field_Iters =50, 
           Sampling_Iters =50)

Outs_2 = Iter_Grabs(Cont_Level= Cont_Levels[1] , 
           Cilantro_df = Cilantro_df, 
           percent_cont =100, 
           Sample_Weight =25,
           N_25g_Samples=N_Samples,
           N_Grabs_Sample =1,
           Plant_Weight = 1,
           loaded_model = qPCR_Model,
           Field_Iters =50, 
           Sampling_Iters =50)

Outs_3 = Iter_Grabs(Cont_Level= Cont_Levels[2] , 
           Cilantro_df = Cilantro_df, 
           percent_cont =100, 
           Sample_Weight =25,
           N_25g_Samples=N_Samples,
           N_Grabs_Sample =1,
           Plant_Weight = 1,
           loaded_model = qPCR_Model,
           Field_Iters =50, 
           Sampling_Iters =50)

Outs_4 = Iter_Grabs(Cont_Level= Cont_Levels[3] , 
           Cilantro_df = Cilantro_df, 
           percent_cont =100, 
           Sample_Weight =25,
           N_25g_Samples=N_Samples,
           N_Grabs_Sample =1,
           Plant_Weight = 1,
           loaded_model = qPCR_Model,
           Field_Iters =50, 
           Sampling_Iters =50)

Outs_5 = Iter_Grabs(Cont_Level= Cont_Levels[4] , 
           Cilantro_df = Cilantro_df, 
           percent_cont =100, 
           Sample_Weight =25,
           N_25g_Samples=N_Samples,
           N_Grabs_Sample =1,
           Plant_Weight = 1,
           loaded_model = qPCR_Model,
           Field_Iters =50, 
           Sampling_Iters =50)

Outs_6 = Iter_Grabs(Cont_Level= Cont_Levels[5] , 
           Cilantro_df = Cilantro_df, 
           percent_cont =100, 
           Sample_Weight =25,
           N_25g_Samples=N_Samples,
           N_Grabs_Sample =1,
           Plant_Weight = 1,
           loaded_model = qPCR_Model,
           Field_Iters =50, 
           Sampling_Iters =50)


All_Grabs_Combined =pd.concat([Outs_1,Outs_2,Outs_3,Outs_4,Outs_5,Outs_6])
All_Grabs_Combined["Conts"] = All_Grabs_Combined["Conts"].astype(str)
All_Grabs_Combined.reset_index(drop= True, inplace= True)





sns.lineplot(data =All_Grabs_Combined, x = "N25gsamples", y = "PDetect", hue ="Conts" )


