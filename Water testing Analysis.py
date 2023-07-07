# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 12:33:55 2022

@author: gareyes3
"""

#%%
import numpy as np
import pandas as pd
import random 
import math
from numpy.random import Generator, PCG64
rng = Generator(PCG64())
import seaborn as sns
import pickle

#%%Functions   
 
def Water_Sampling (total_oocyst_bw, bw_volume, sample_size_volume,total_samples, loaded_model):
    sample_resuls =[]
    for i in range(total_samples):
        Pr_Ocyst_BW= sample_size_volume /(bw_volume)#probability of finding an ocyst in sample water
    
        #total Ocyst that actually made it to our sample
        T_Ocyst_SBW = np.random.binomial(n = total_oocyst_bw, p =Pr_Ocyst_BW) #SBW = sample bulk water
        #print(T_Ocyst_SBW)
        #Total Occyst recovered
        #T_Ocyst_Rec_SBW = rng.binomial(n=T_Ocyst_SBW, p =filter_recovery)
        # PCR Confirmation 
        if T_Ocyst_SBW>=1:
            Pr_Detect = loaded_model.predict_proba(np.array([T_Ocyst_SBW]).reshape(-1,1))[0][1] #from logistic
        else:
            Pr_Detect = 0
        if np.random.uniform(0,1) < Pr_Detect:
            sample_resuls.append(1)
        else:
            sample_resuls.append(0)
    if sum(sample_resuls)> 0:
        return 1
    else:
        return 0
   
#%%
#%%
#Loading the saved logistic regression model

#filename = 'C:\\Users\gareyes3\Documents\GitHub\CPS-Farm-To-Facility-Cilantro\logistic_Prod_Test.sav'
filename_qPCR = 'C:\\Users\gareyes3\Documents\GitHub\CPS-Farm-To-Facility-Cilantro\logistic_AW_Testing_qPCR.sav'
qPCR_Model_AW = pickle.load(open(filename_qPCR, 'rb'))

#%%

def Iterating_Water_Samples (total_oocyst_bw, bw_volume, sample_size_volume, total_samples,loaded_model,sampling_iters):
    P_detect = []
    for i in range(sampling_iters):
        P_detect.append(Water_Sampling (total_oocyst_bw=total_oocyst_bw, 
                                        bw_volume=bw_volume, 
                                        sample_size_volume =sample_size_volume,
                                        total_samples=total_samples,
                                        loaded_model =loaded_model ))
    return np.mean(P_detect)


#Water Characteristics, not necesarily used. 
Water_Irrigation_In = 12 #Inches of water per harvest season
Total_L_Season = 40.46*40.46*(0.0254*Water_Irrigation_In)*1000 # one acre 40.46m2 * 0.348 m of water * 1000 to convert tot L
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



def Iterating_Water_Samples_Uncert (total_oocyst_bw, bw_volume, sample_size_volume,total_samples, loaded_model,sampling_iters, var_iters):
    Ps_detect=[]
    for i in range(var_iters):
        Ps_detect.append(Iterating_Water_Samples (total_oocyst_bw = total_oocyst_bw, 
                                         bw_volume =bw_volume , 
                                         sample_size_volume=sample_size_volume ,
                                         total_samples=total_samples,
                                         loaded_model =loaded_model ,
                                         sampling_iters =sampling_iters))
    return Ps_detect






def Analysis_NSamples (Cont_Levels, bw_volume, sample_size_volume,total_samples, loaded_model,sampling_iters, var_iters):
    Df_Outs = pd.DataFrame(columns=["PDetect", "Conts", "N10Lsamples"]) #creating empty collection dataframe
    for i in Cont_Levels:
        print(i)
        Cont_BW = bw_volume*i
        P_Detect_L= Iterating_Water_Samples_Uncert  (total_oocyst_bw = Cont_BW , 
                                                     bw_volume =bw_volume , 
                                                     sample_size_volume=sample_size_volume , 
                                                     total_samples= total_samples,
                                                     loaded_model =loaded_model ,
                                                     sampling_iters =sampling_iters,
                                                     var_iters =var_iters)
    
        Df_Outs2 = pd.DataFrame({
            "PDetect":P_Detect_L,
            "Conts":i,
            "N10Lsamples":np.repeat(total_samples,var_iters)
            })
        Df_Outs= pd.concat([Df_Outs,Df_Outs2])
    return Df_Outs

#%%

Cont_Levels = list(np.arange(0,2.01,0.01)) #Oocyst per L

#%%

Cont_Levels = list(np.arange(0,3,0.01)) #Oocyst per L

Sample_BW_1 =Analysis_NSamples (Cont_Levels =Cont_Levels, 
                   bw_volume =Total_L_Season, 
                   sample_size_volume =10,
                   total_samples =1, 
                   loaded_model =qPCR_Model_AW,
                   sampling_iters =1000, 
                   var_iters=1)

Sample_BW_1.to_csv("C:\\Users\\gareyes3\\Documents\\GitHub\\CPS-Farm-To-Facility-Cilantro\\Data_Cilantro_Outputs\\Water_Testing_Meds.csv")


#%%
Sample_BW_1 =Analysis_NSamples (Cont_Levels =Cont_Levels, 
                   bw_volume =Total_L_Season, 
                   sample_size_volume =10,
                   total_samples =1, 
                   loaded_model =qPCR_Model_AW,
                   sampling_iters =100, 
                   var_iters=100)

Sample_BW_2 =Analysis_NSamples (Cont_Levels =Cont_Levels, 
                   bw_volume =Total_L_Season, 
                   sample_size_volume =10,
                   total_samples =2, 
                   loaded_model =qPCR_Model_AW,
                   sampling_iters =100, 
                   var_iters=100)

Sample_BW_4 =Analysis_NSamples (Cont_Levels =Cont_Levels, 
                   bw_volume =Total_L_Season, 
                   sample_size_volume =10,
                   total_samples =4, 
                   loaded_model =qPCR_Model_AW,
                   sampling_iters =100, 
                   var_iters=100)

Sample_BW_8 =Analysis_NSamples (Cont_Levels =Cont_Levels, 
                   bw_volume =Total_L_Season, 
                   sample_size_volume =10,
                   total_samples =8, 
                   loaded_model =qPCR_Model_AW,
                   sampling_iters =100, 
                   var_iters=100)

Sample_BW_16 =Analysis_NSamples (Cont_Levels =Cont_Levels, 
                   bw_volume =Total_L_Season, 
                   sample_size_volume =10,
                   total_samples =16, 
                   loaded_model =qPCR_Model_AW,
                   sampling_iters =100, 
                   var_iters=100)

Sample_BW_32 =Analysis_NSamples (Cont_Levels =Cont_Levels, 
                   bw_volume =Total_L_Season, 
                   sample_size_volume =10,
                   total_samples =32, 
                   loaded_model =qPCR_Model_AW,
                   sampling_iters =100, 
                   var_iters=100)

All_Samples_Combined =pd.concat([Sample_BW_1,Sample_BW_2,Sample_BW_4,Sample_BW_8,Sample_BW_16,Sample_BW_32])
All_Samples_Combined["N10Lsamples"] = All_Samples_Combined["N10Lsamples"].astype(str)
All_Samples_Combined.reset_index(drop= True, inplace= True)

All_Samples_Combined.to_csv("C:\\Users\\gareyes3\\Documents\\GitHub\\CPS-Farm-To-Facility-Cilantro\\Data_Cilantro_Outputs\\Water_Testing_Analysis_R4.csv")

sns.lineplot(data =All_Samples_Combined, x = "Conts", y = "PDetect", hue ="N10Lsamples" )
plt.xlim(0,5)