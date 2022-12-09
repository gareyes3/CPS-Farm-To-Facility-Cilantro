# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:12:21 2022

@author: gareyes3
"""
import pandas as pd
import numpy as np
import math
import random
import pickle

#Important Functions
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
            Contamination_Pattern = np.random.multinomial(Hazard_lvl_percluster,[1/No_Cont_PartitionUnits]*No_Cont_PartitionUnits,1) #spliting cont into chunks length
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
        Pr_Detect = loaded_model.predict_proba(np.array([Total_Oo_Composite]).reshape(-1,1))[0][1] #from logistic
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

def Water_Sampling (total_oocyst_bw, bw_volume, sample_size_volume,total_samples, loaded_model):
    sample_resuls =[]
    for i in range(total_samples):
        Pr_Ocyst_BW= sample_size_volume /(bw_volume)#probability of finding an ocyst in sample water
    
        #total Ocyst that actually made it to our sample
        T_Ocyst_SBW = np.random.binomial(n = total_oocyst_bw, p =Pr_Ocyst_BW) #SBW = sample bulk water
    
        #Total Occyst recovered
        #T_Ocyst_Rec_SBW = rng.binomial(n=T_Ocyst_SBW, p =filter_recovery)
        # PCR Confirmation 
        Pr_Detect = loaded_model.predict_proba(np.array([T_Ocyst_SBW]).reshape(-1,1))[0][1] #from logistic
        if np.random.uniform(0,1) < Pr_Detect:
            sample_resuls.append(1)
        else:
            sample_resuls.append(0)
    if sum(sample_resuls)> 0:
        return 1
    else:
        return 0


#%% Loading PCR Detection Models
filename_qPCR = 'C:\\Users\gareyes3\Documents\GitHub\CPS-Farm-To-Facility-Cilantro\logistic_AW_Testing_qPCR.sav'
qPCR_Model_AW = pickle.load(open(filename_qPCR, 'rb'))
#%%

#Cilantro  Inputs

#Creating the Field
Field_Yield = 22_000 #lb
Plant_Weight = 1 #lb
Total_Plants = int(Field_Yield/Plant_Weight)
Total_Plants_List = range(1,Total_Plants+1)

Case_Weight = 20 #lb per case
Bunches_Weight = Case_Weight/Plant_Weight

#Water and Season Characteristics
Water_Irrigation_In = 12 #Inches of water per harvest season
Total_L_Season = 40.46*40.46*(0.0254*Water_Irrigation_In)*1000 # one acre 40.46m2 * 0.348 m of water * 10000 to convert tot m3
Days_per_season = 45 #days
L_water_day = Total_L_Season/Days_per_season

#Contamination
Per_Cont_Field = 100


#Water contamintion to calculate min and max. 
OO_per_L = 0.6
OO_per_H = 20
OO_per_Input = 1

Initial_Levels_Bulk_Low = int(Total_L_Season*OO_per_L)
Initial_Levels_Bulk_Day_Low = L_water_day*OO_per_L
#299376, Oo per season
#6652.8, Oo per day
Initial_Levels_Bulk_High = int(Total_L_Season*OO_per_H)
Initial_Levels_Bulk_Day_High = L_water_day*OO_per_H

#Initial levels based on Input
Initial_Levels_Bulk = int(Total_L_Season*OO_per_Input)

#Water Sampling
Sampling_every_Days_Water = 3
Water_Sampling_Days = np.arange(1,Days_per_season,Sampling_every_Days_Water)

#Water Sampling
W_Sample_Vol = 10 #L
Total_Samples_Day = 1
#%%
# Process Model --------------------------------------------------------------

#Creating field dataframe
Cilantro_df=pd.DataFrame({"Plant_ID": Total_Plants_List,
                       "Weight": Plant_Weight,
                       "Case_PH": 0,
                       "Oo": 0,
                       "Oo_BRej":"",
                       "Location": 1,
                       'PositiveSamples':0,
                       "Rej_Acc" :"Acc"
                  })

for i in range (1,Days_per_season+1):
    
    #Water Sampling: Happens in sampling days
    if i in Water_Sampling_Days:
        W_Test_Outcome =W_Detected_YN = Water_Sampling (total_oocyst_bw = Initial_Levels_Bulk, 
                                        bw_volume =Total_L_Season , 
                                        sample_size_volume =W_Sample_Vol,
                                        total_samples = Total_Samples_Day, 
                                        loaded_model =qPCR_Model_AW )
        print(W_Test_Outcome)
    
    #Condition here, what happens if we detect cyclospora in water sample? 
        #- if water test positive and it is in the test day we do not irrigate
        #- if water test negative or if we are not on a sampling day we irrigate
    if W_Test_Outcome == 0 or i not in Water_Sampling_Days:
        #Step 1: Irrigation Event
        Cilantro_df = field_cont_percetage2(df = Cilantro_df, 
                                            percent_cont = Per_Cont_Field, 
                                            Hazard_lvl =Initial_Levels_Bulk_Day_Low ,
                                            No_Cont_Clusters = 1)
        
    elif W_Test_Outcome == 1 and i in Water_Sampling_Days:
        Cilantro_df = field_cont_percetage2(df = Cilantro_df, 
                                            percent_cont = Per_Cont_Field, 
                                            Hazard_lvl =0,
                                            No_Cont_Clusters = 1)
    
    
    
        
        
        
        
        
    
    
    
    
    
    
    

