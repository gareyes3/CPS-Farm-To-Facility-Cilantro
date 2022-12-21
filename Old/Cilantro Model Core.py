# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 09:12:20 2022

@author: gareyes3
"""
#%%
import numpy as np
import pandas as pd
import random 
import math
from numpy.random import Generator, PCG64
rng = Generator(PCG64())
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit, cuda
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
    T_Ocyst_SBW = rng.binomial(n = total_oocyst_bw, p =Pr_Ocyst_BW) #SBW = sample bulk water
    #print(T_Ocyst_SBW)
    #Total Occyst recovered
    T_Ocyst_Rec_SBW = rng.binomial(n=T_Ocyst_SBW, p =filter_recovery)
    # Sample Results. 
    Total_Confirmed =[]
    for k in range(1,T_Ocyst_Rec_SBW+1):
        if random.uniform(0,1) <Pr_PCR_Con:
            Total_Confirmed.append(k)
    
    if len(Total_Confirmed)>1:
        return 1
    else:
        return 0
    

    
#%%
#Creating the Field
Field_Yield = 22_000 #lb
Plant_Weight = 1 #lb
Total_Plants = int(Field_Yield/Plant_Weight)
Total_Plants_List = range(1,Total_Plants+1)

Case_Weight = 20 #lb per case
Bunches_Weight = Case_Weight/Plant_Weight


Cilantro_df=pd.DataFrame({"Plant_ID": Total_Plants_List,
                       "Weight": Plant_Weight,
                       "Case_PH": 0,
                       "Oo": 0,
                       "Oo_BRej":"",
                       "Location": 1,
                       'PositiveSamples':0,
                       "Rej_Acc" :"Acc"
                  })

#Scenario Params
Water_testing_YN = True


#Water characteristiscs
Water_Irrigation_In = 12 #Inches
Total_L_Season = 40.46*40.46*0.348*1000 # one acre 40.46m2 * 0.348 m of water * 10000 to convert tot m3
Days_per_season = 45 #days
L_water_day = Total_L_Season/Days_per_season

 

Scenario1_oo_l_l = 0.6 #oocist per liter scenario 1
Scenario1_oo_h_l = 2.5 #oocyst per liter scenario 1 high
Scenario2_oo_l = 20  # oocyst per liter scenario 2

# for field contamination

#safety parameters
water_test_freq = 3 #water testign every 3 days. 

#Step 1: water testing. 

#water testing, every so many days. 
#we are assuming total bulk water is the total water that will esed in a harvest season
#Defining total Oocyst contamination
Initial_Levels_Bulk = int(Total_L_Season*Scenario1_oo_l_l)
#List_of_OOc = range(1,Initial_Levels+1)

Irrigation_Levels_Day = L_water_day*Scenario1_oo_l_l


#Defining sampling plan characteristics
Sample_Size_BW_G = 10 #Total 10 gallon sample size.
Sample_Size_BW = Sample_Size_BW_G*3.74 #total Liters. 

Pr_Rec_Filter_1623 = 0.16 #method 1623
Pr_Rec_Filter_Duf= 0.17 #method 1623
Pr_Rec_Filter_FDA= 10.7 #method 1623

Pr_PCR_Con = 0.98

#Mass for
N_Grabs = 60
Composite_Mass = 375
grabs_weight = (375/60)/454



#begining of day water testing

if Water_testing_YN == True:
    Detect =Water_Sampling (total_oocyst_bw =Initial_Levels_Bulk , 
                    bw_volume =Total_L_Season, 
                    sample_size_volume =Sample_Size_BW, 
                    filter_recovery =Pr_Rec_Filter_1623, 
                    PCR_confirmation =Pr_PCR_Con)
    print(Detect)
    
    if Detect == 1:
        Cilantro_df = field_cont_percetage2(df = Cilantro_df,
                                            percent_cont =100, 
                                            Hazard_lvl =0,
                                            No_Cont_Clusters =1)
    if Detect == 0:
        Cilantro_df = field_cont_percetage2(df = Cilantro_df,
                                            percent_cont =100, 
                                            Hazard_lvl =Irrigation_Levels_Day,
                                            No_Cont_Clusters =1)
        
#Finished product testing

Oo_list=Cilantro_df.loc[:,"Oo"]
for j in range(N_Grabs):
    List_Random=Oo_list.sample(n=1)
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
    







Initial_Levels_Bulk = int(Total_L_Season*Scenario1_oo_l_l)
        
    
#FDA recovery rate "validation"
rec_rate = list(np.arange(0, 1, 0.001))

det_rate_list = []
for j in rec_rate :
    sampling_results = []
    for i in range(1000):
        Detect =Water_Sampling (total_oocyst_bw =Initial_Levels_Bulk , 
                        bw_volume =Total_L_Season, 
                        sample_size_volume =Sample_Size_BW, 
                        filter_recovery =j, 
                        PCR_confirmation =Pr_PCR_Con)
        
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
plt.title("Detection vs Filter Recovery Rate 6 oocyst/ 10L sample")

df_rec_ra=pd.DataFrame({"rr": rec_rate,
                        "dt":det_rate_list})


Probs_detect = []
for i in Initial_Levels:
    Detect =[]
    for j in list(range(1000)):
        Total_Cont_BW = i #oocyst #BW is bulk water
        Total_BW = 1_000_000 #liters
        Sample_Size_BW = 50*3.74 #liters
        
        #Water sampling M1: 
        Pr_Ocyst_BW= Sample_Size_BW /Total_BW #probability of finding an ocyst in sample water
        
        #total Ocyst that actually made it to our sample
        T_Ocyst_SBW = np.random.binomial(n = Total_Cont_BW, p =Pr_Ocyst_BW) #SBW = sample bulk water
        
        #Recovery rate. 
        Pr_Rec_Filter = 0.16
        
        #Total Occyst recovered
        T_Ocyst_Rec_SBW = np.random.binomial(n=T_Ocyst_SBW, p =Pr_Rec_Filter)
        
        #If there is a pcr confirmation %
        Pr_PCR_Con = 0.997
        
        Total_Confirmed =[]
        for k in range(1,T_Ocyst_Rec_SBW+1):
            if random.uniform(0,1) <Pr_PCR_Con:
                Total_Confirmed.append(k)
        
        if len(Total_Confirmed)>1:
            Detect.append(1) 
        else:
            Detect.append(0) 
    Pr_Detect = sum(Detect)/1000
    Probs_detect.append(Pr_Detect)
    

    


sns.lineplot(x=Initial_Levels ,y=Probs_detect)


Initial_Levels[295]

np.log10(295_000)

295_000/1_000_00



