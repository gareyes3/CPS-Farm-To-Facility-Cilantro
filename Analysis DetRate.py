# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 18:47:01 2022

@author: Gustavo Reyes
"""

#%%
import sys
sys.path
sys.path.append('C:\\Users\Gustavo Reyes\Documents\GitHubFiles\CPS-Farm-To-Facility-Cilantro')
sys.path.append('C:\\Users\gareyes3\Documents\GitHub\CPS-Farm-To-Facility-Cilantro')
sys.path.append('C:\\Users\\reyes\\Documents\\GitHub\\CPS-Farm-To-Facility-Cilantro')
#%%
import numpy as np
import pandas as pd
import random 
import math
from numpy.random import Generator, PCG64
rng = Generator(PCG64())
import seaborn as sns
#%%Functions
def random_chunk(lst, chunk_size):
    nb_chunks = int(math.ceil(len(lst)/chunk_size))
    choice = random.randrange(nb_chunks) # 0 <= choice < nb_chunks
    return lst[choice*chunk_size:(choice+1)*chunk_size]

def field_cont_percetage2(df, percent_cont, Hazard_lvl,No_Cont_Clusters):
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
            
        df.update(Field_df_1)
        return df
    
def Water_Sampling (total_oocyst_bw, bw_volume, sample_size_volume, filter_recovery, PCR_confirmation):
    Pr_Ocyst_BW= sample_size_volume /bw_volume #probability of finding an ocyst in sample water
    #total Ocyst that actually made it to our sample
    T_Ocyst_SBW = np.random.binomial(n = total_oocyst_bw, p =Pr_Ocyst_BW) #SBW = sample bulk water
    #print(T_Ocyst_SBW)
    #Total Occyst recovered
    T_Ocyst_Rec_SBW = np.random.binomial(n=T_Ocyst_SBW, p =filter_recovery)
    # Sample Results. 
    Total_Confirmed =[]
    for k in range(1,T_Ocyst_Rec_SBW+1):
        if random.uniform(0,1) <PCR_confirmation:
            Total_Confirmed.append(k)
    if len(Total_Confirmed)>1:
        return 1
    else:
        return 0
    
    
#%%

#Initial Mass Paramters
Field_Yield = 22_000 #lb
Plant_Weight = 1 #lb
Total_Plants = int(Field_Yield/Plant_Weight)
Total_Plants_List = range(1,Total_Plants+1) 

#Packing product parameters:
Case_Weight = 20 #lb per case
Bunches_Weight = Case_Weight/Plant_Weight

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
Water_Test_Frequency = 3 #Every 3 days
Sample_Size_BW_G = 10 #Total 10 gallon sample size.
Sample_Size_BW_L = 10#Sample_Size_BW_G*3.74 #total Liters. 
Pr_Rec_Filter_1623 = 0.16 #method 1623
Pr_Rec_Filter_Duf= 0.17 #method 1623
Pr_Rec_Filter_FDA= 0.39 #method 1623
Pr_PCR_Con = 0.98 #PCR Confirmation

#Sampling Characteristics Product
N_Grabs = 60
Composite_Mass = 375
grabs_weight = (Composite_Mass/N_Grabs)/454 #in lbs. 

#%% Water Testing Model
#Uncertatinty Water Contamination. 

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

    
Output_1623 = Calc_Pdetect_From_Cont_Levels(Cont_Levels=Cont_Levels , 
                              Total_L_Season= Total_L_Season , 
                              Sample_Size_BW_L = Sample_Size_BW_L , 
                              filter_recovery= Pr_Rec_Filter_1623,
                              Pr_PCR_Con =  Pr_PCR_Con,
                              total_Probdet_iters= 100,
                              unc_iters= 100)

Output_Duf = Calc_Pdetect_From_Cont_Levels(Cont_Levels=Cont_Levels , 
                              Total_L_Season= Total_L_Season , 
                              Sample_Size_BW_L = Sample_Size_BW_L , 
                              filter_recovery= Pr_Rec_Filter_Duf,
                              Pr_PCR_Con =  Pr_PCR_Con,
                              total_Probdet_iters= 100,
                              unc_iters= 100)

Output_FDA = Calc_Pdetect_From_Cont_Levels(Cont_Levels=Cont_Levels , 
                              Total_L_Season= Total_L_Season , 
                              Sample_Size_BW_L = Sample_Size_BW_L , 
                              filter_recovery= Pr_Rec_Filter_FDA,
                              Pr_PCR_Con =  Pr_PCR_Con,
                              total_Probdet_iters= 100,
                              unc_iters= 100)

methods_combined = pd.concat([Output_1623,Output_Duf,Output_FDA])
methods_combined["Rec_Rate"] = methods_combined["Rec_Rate"].astype(str)

methods_combined.to_csv(path_or_buf = "C:\\Users\\gareyes3\\Documents\\GitHub\\CPS-Farm-To-Facility-Cilantro\\Data_Cilantro_Outputs\\Contamination_Unc.csv")

#%%
#Now we can proceeed and get the optimization part. Bulk water. 

Cont_Levels = list(np.arange(0,30, 0.1)) #oocyst per liter. 

 
#Method 1623
       
Output_1G = Calc_Pdetect_From_Cont_Levels(Cont_Levels=Cont_Levels , 
                              Total_L_Season= Total_L_Season , 
                              Sample_Size_BW_L = 1,#Sample_Size_BW_L , 
                              filter_recovery= Pr_Rec_Filter_1623,
                              Pr_PCR_Con =  Pr_PCR_Con,
                              total_Probdet_iters= 100,
                              unc_iters= 100)

Output_1G["Volume"] = "1 Liter(s)"

Output_5G = Calc_Pdetect_From_Cont_Levels(Cont_Levels=Cont_Levels , 
                              Total_L_Season= Total_L_Season , 
                              Sample_Size_BW_L =5,#Sample_Size_BW_L , 
                              filter_recovery= Pr_Rec_Filter_1623,
                              Pr_PCR_Con =  Pr_PCR_Con,
                              total_Probdet_iters= 100,
                              unc_iters= 100)

Output_5G["Volume"] = "5 Liter(s)"

Output_10G = Calc_Pdetect_From_Cont_Levels(Cont_Levels=Cont_Levels , 
                              Total_L_Season= Total_L_Season , 
                              Sample_Size_BW_L = 10,#Sample_Size_BW_L , 
                              filter_recovery= Pr_Rec_Filter_1623,
                              Pr_PCR_Con =  Pr_PCR_Con,
                              total_Probdet_iters= 100,
                              unc_iters= 100)

Output_10G["Volume"] = "10 Liter(s)"

Output_50G = Calc_Pdetect_From_Cont_Levels(Cont_Levels=Cont_Levels , 
                              Total_L_Season= Total_L_Season , 
                              Sample_Size_BW_L = 50,#Sample_Size_BW_L , 
                              filter_recovery= Pr_Rec_Filter_1623,
                              Pr_PCR_Con =  Pr_PCR_Con,
                              total_Probdet_iters= 100,
                              unc_iters= 100)

Output_50G["Volume"] = "50 Liter(s)"


#Method DUF ----------------------------------------------------------

Output_1G_DUF = Calc_Pdetect_From_Cont_Levels(Cont_Levels=Cont_Levels , 
                              Total_L_Season= Total_L_Season , 
                              Sample_Size_BW_L = 1,#Sample_Size_BW_L , 
                              filter_recovery= Pr_Rec_Filter_Duf,
                              Pr_PCR_Con =  Pr_PCR_Con,
                              total_Probdet_iters= 100,
                              unc_iters= 100)

Output_1G_DUF["Volume"] = "1 Liter(s)"

Output_5G_DUF = Calc_Pdetect_From_Cont_Levels(Cont_Levels=Cont_Levels , 
                              Total_L_Season= Total_L_Season , 
                              Sample_Size_BW_L =5,#Sample_Size_BW_L , 
                              filter_recovery= Pr_Rec_Filter_Duf,
                              Pr_PCR_Con =  Pr_PCR_Con,
                              total_Probdet_iters= 100,
                              unc_iters= 100)

Output_5G_DUF["Volume"] = "5 Liter(s)"

Output_10G_DUF= Calc_Pdetect_From_Cont_Levels(Cont_Levels=Cont_Levels , 
                              Total_L_Season= Total_L_Season , 
                              Sample_Size_BW_L = 10,#Sample_Size_BW_L , 
                              filter_recovery= Pr_Rec_Filter_Duf,
                              Pr_PCR_Con =  Pr_PCR_Con,
                              total_Probdet_iters= 100,
                              unc_iters= 100)

Output_10G_DUF["Volume"] = "10 Liter(s)"

Output_50G_DUF = Calc_Pdetect_From_Cont_Levels(Cont_Levels=Cont_Levels , 
                              Total_L_Season= Total_L_Season , 
                              Sample_Size_BW_L = 50,#Sample_Size_BW_L , 
                              filter_recovery= Pr_Rec_Filter_Duf,
                              Pr_PCR_Con =  Pr_PCR_Con,
                              total_Probdet_iters= 100,
                              unc_iters= 100)

Output_50G_DUF["Volume"] = "50 Liter(s)"


#Method FDA ----------------------------------------------------------

Output_1G_FDA = Calc_Pdetect_From_Cont_Levels(Cont_Levels=Cont_Levels , 
                              Total_L_Season= Total_L_Season , 
                              Sample_Size_BW_L = 1,#Sample_Size_BW_L , 
                              filter_recovery= Pr_Rec_Filter_FDA,
                              Pr_PCR_Con =  Pr_PCR_Con,
                              total_Probdet_iters= 100,
                              unc_iters= 100)

Output_1G_FDA["Volume"] = "1 Liter(s)"

Output_5G_FDA = Calc_Pdetect_From_Cont_Levels(Cont_Levels=Cont_Levels , 
                              Total_L_Season= Total_L_Season , 
                              Sample_Size_BW_L =5,#Sample_Size_BW_L , 
                              filter_recovery= Pr_Rec_Filter_FDA,
                              Pr_PCR_Con =  Pr_PCR_Con,
                              total_Probdet_iters= 100,
                              unc_iters= 100)

Output_5G_FDA["Volume"] = "5 Liter(s)"

Output_10G_FDA= Calc_Pdetect_From_Cont_Levels(Cont_Levels=Cont_Levels , 
                              Total_L_Season= Total_L_Season , 
                              Sample_Size_BW_L = 10,#Sample_Size_BW_L , 
                              filter_recovery= Pr_Rec_Filter_FDA,
                              Pr_PCR_Con =  Pr_PCR_Con,
                              total_Probdet_iters= 100,
                              unc_iters= 100)

Output_10G_FDA["Volume"] = "10 Liter(s)"

Output_50G_FDA = Calc_Pdetect_From_Cont_Levels(Cont_Levels=Cont_Levels , 
                              Total_L_Season= Total_L_Season , 
                              Sample_Size_BW_L = 50,#Sample_Size_BW_L , 
                              filter_recovery= Pr_Rec_Filter_FDA,
                              Pr_PCR_Con =  Pr_PCR_Con,
                              total_Probdet_iters= 100,
                              unc_iters= 100)

Output_50G_FDA["Volume"] = "50 Liter(s)"



methods_combined_Vol = pd.concat([Output_1G,Output_5G,Output_10G,Output_50G,
                                  Output_1G_DUF,Output_5G_DUF,Output_10G_DUF,Output_50G_DUF,
                                  Output_1G_FDA,Output_5G_FDA,Output_10G_FDA,Output_50G_FDA])
methods_combined_Vol["Rec_Rate"] = methods_combined_Vol["Rec_Rate"].astype(str)

methods_combined_Vol.to_csv(path_or_buf = "C:\\Users\\gareyes3\\Documents\\GitHub\\CPS-Farm-To-Facility-Cilantro\\Data_Cilantro_Outputs\\Volume_Unc.csv")




