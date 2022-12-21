# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 10:25:28 2022

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
 
def Water_Sampling (total_oocyst_bw, bw_volume, sample_size_volume, filter_recovery, loaded_model):
    Pr_Ocyst_BW= sample_size_volume /(bw_volume)#probability of finding an ocyst in sample water
    #total Ocyst that actually made it to our sample
    T_Ocyst_SBW = rng.binomial(n = total_oocyst_bw, p =Pr_Ocyst_BW) #SBW = sample bulk water
    #Total Occyst recovered
    T_Ocyst_Rec_SBW = rng.binomial(n=T_Ocyst_SBW, p =filter_recovery)
    # PCR Confirmation 
    Pr_Detect = loaded_model.predict_proba(np.array([T_Ocyst_Rec_SBW]).reshape(-1,1))[0][1] #from logistic
    if np.random.uniform(0,1) < Pr_Detect:
        return 1
    else:
        return 0
    
#%%
filename = 'C:\\Users\Gustavo Reyes\Documents\GitHubFiles\CPS-Farm-To-Facility-Cilantro\logistic_Prod_Test.sav'
loaded_model = pickle.load(open(filename, 'rb'))

#Water Characteristics
Water_Irrigation_In = 12 #Inches of water per harvest season
Total_L_Season = 40.46*40.46*(0.0254*Water_Irrigation_In)*1000 # one acre 40.46m2 * 0.348 m of water * 10000 to convert tot m3
Days_per_season = 45 #days
L_water_day = Total_L_Season/Days_per_season

#Water contamintion
OO_per_L = 0.6
Initial_Levels_Bulk = int(Total_L_Season*OO_per_L)
Initial_Levels_Bulk_Day = L_water_day*OO_per_L


#Sampling Characteristics Water
Sample_Size_BW_L = 10#Sample_Size_BW_G*3.74 #total Liters. 
Pr_PCR_Con = 0.98 #PCR Confirmation


#uncertainty around recovery rates:
rec_rate = list(np.arange(0, 1, 0.001))

det_rate_list = []
for j in rec_rate :
    print(j)
    sampling_results = []
    for i in range(1000):
        Detect =Water_Sampling (total_oocyst_bw =Initial_Levels_Bulk , 
                        bw_volume =Total_L_Season, 
                        sample_size_volume =Sample_Size_BW_L, 
                        filter_recovery =j, 
                        loaded_model =loaded_model)
        
        sampling_results.append(Detect)
    
    zeros = []
    for k in sampling_results:
        if k == 1:
         zeros.append(1)  
    
    det_rate =len(zeros)/1000
    det_rate_list.append(det_rate)
    
sns.lineplot(x=rec_rate ,y=det_rate_list)
plt.xlabel("Filter Recovery Rate")
plt.ylabel("Detection Rate")
plt.title("Detection vs Filter Recovery Rate 6 oocyst/10L sample")

df_rec_rate=pd.DataFrame({"rr": rec_rate,
                        "dt":det_rate_list})

df_rec_rate.to_csv(path_or_buf = "C:\\Users\\gareyes3\\Documents\\GitHub\\CPS-Farm-To-Facility-Cilantro\\Data_Cilantro_Outputs\\RR_FDA_Validation.csv")



#Recovery rate for FDA = 0.39

#Validation for other method. expcted result should be 1.


def Calc_Prob_Detection (total_oocyst_bw, bw_volume,sample_size_volume,filter_recovery, PCR_confirmation, tot_iters):
    Detection_YN= []
    for i in range(tot_iters):
        Result = Water_Sampling (total_oocyst_bw, 
                                 bw_volume, 
                                 sample_size_volume, 
                                 filter_recovery, 
                                 PCR_confirmation)
        Detection_YN.append(Result)
    return np.mean(Detection_YN)

#Comparisons = 
    #1.Contamination Levels
    #2.Method (Recovery Rate from Filter)
    

def Calc_Pdetect_From_Cont_Levels(Cont_Levels, Total_L_Season, Sample_Size_BW_L, filter_recovery,Pr_PCR_Con, total_Probdet_iters, unc_iters):
    Output2 = pd.DataFrame(columns=["Contamination_Level", "PDetect", "Rec_Rate"])
    for j in range(unc_iters):
        print(j)
        Prob_Dets= []
        for i in Cont_Levels:  
            Initial_Levels_Bulk = int(Total_L_Season*i)
            Prob_det = Calc_Prob_Detection(total_oocyst_bw= Initial_Levels_Bulk,
                                            bw_volume = Total_L_Season,
                                            sample_size_volume = Sample_Size_BW_L,
                                            filter_recovery = filter_recovery,
                                            PCR_confirmation =Pr_PCR_Con,
                                            tot_iters= total_Probdet_iters)
            
            Prob_Dets.append(Prob_det)  
        
        Output= pd.DataFrame({
            "Contamination_Level": Cont_Levels,
            "PDetect": Prob_Dets,
            "Rec_Rate": filter_recovery
            })
    
        Output2 =pd.concat([Output2,Output])
    
    return (Output2)   

#Create a list
# [0.6, 2.5, 20] Range based on the literature
Cont_Levels = list(np.arange(0,20, 0.1)) #oocyst per liter. 

    
Output_validation = Calc_Pdetect_From_Cont_Levels(Cont_Levels=Cont_Levels , 
                              Total_L_Season= Total_L_Season , 
                              Sample_Size_BW_L = Sample_Size_BW_L , 
                              filter_recovery= 0.39,
                              Pr_PCR_Con =  Pr_PCR_Con,
                              total_Probdet_iters= 100,
                              unc_iters= 100)



Output_validation.to_csv(path_or_buf = "C:\\Users\\gareyes3\\Documents\\GitHub\\CPS-Farm-To-Facility-Cilantro\\Data_Cilantro_Outputs\\RR_FDA_Validation_Conts.csv")



#################

#25 oocyst per L
OO_per_L = 2.5
Initial_Levels_Bulk = int(Total_L_Season*OO_per_L)

sampling_results = []
for i in range(1000):
    Detect =Water_Sampling (total_oocyst_bw =Initial_Levels_Bulk , 
                    bw_volume =Total_L_Season, 
                    sample_size_volume =Sample_Size_BW_L, 
                    filter_recovery =0.39, 
                    PCR_confirmation =Pr_PCR_Con)
    
    sampling_results.append(Detect)

np.mean(sampling_results)

#200 Oocyst per L
OO_per_L = 20
Initial_Levels_Bulk = int(Total_L_Season*OO_per_L)

sampling_results = []
for i in range(1000):
    Detect =Water_Sampling (total_oocyst_bw =Initial_Levels_Bulk , 
                    bw_volume =Total_L_Season, 
                    sample_size_volume =Sample_Size_BW_L, 
                    filter_recovery =0.39, 
                    PCR_confirmation =Pr_PCR_Con)
    
    sampling_results.append(Detect)

np.mean(sampling_results)
