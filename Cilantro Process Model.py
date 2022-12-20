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
    probs_detection = []
    sample_resuls =[]
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
        probs_detection.append(Pr_Detect)
        if np.random.uniform(0,1) < Pr_Detect:
            sample_resuls.append(1)
            df2.loc[Sample_Indeces, 'PositiveSamples'] = df2.loc[Sample_Indeces, 'PositiveSamples'] + 1
        else: 
            sample_resuls.append(0)        
    if sum(sample_resuls)> 0:
        reject_YN = 1
    else:
       reject_YN = 0
        
    if len(probs_detection)>1:
        pdetect = 1-np.prod([1-i for i in probs_detection])
    else: 
        pdetect = probs_detection[0]
    return [df2, reject_YN,pdetect]


def F_Rejection_Rule_C (df):
    df_field_1 =df.copy()
    Postives = sum(df_field_1['PositiveSamples'] >0)
    if Postives>0:
     df_field_1.loc[:,"Rej_Acc"] = "REJ"
     df_field_1.loc[:,"Oo_BRej"] = df_field_1["Oo"]
     df_field_1.loc[:,"Oo"] = 0
    return df_field_1

def Func_Water_Sampling (total_oocyst_bw, bw_volume, sample_size_volume,total_samples, loaded_model):
    sample_resuls =[]
    probs_detection = []
    for i in range(total_samples):
        Pr_Ocyst_BW= sample_size_volume /(bw_volume)#probability of finding an ocyst in sample water
    
        #total Ocyst that actually made it to our sample
        T_Ocyst_SBW = np.random.binomial(n = total_oocyst_bw, p =Pr_Ocyst_BW) #SBW = sample bulk water
    
        #Total Occyst recovered
        #T_Ocyst_Rec_SBW = rng.binomial(n=T_Ocyst_SBW, p =filter_recovery)
        # PCR Confirmation 
        Pr_Detect = loaded_model.predict_proba(np.array([T_Ocyst_SBW]).reshape(-1,1))[0][1] #from logistic
        probs_detection.append(Pr_Detect)
        if np.random.uniform(0,1) < Pr_Detect:
            sample_resuls.append(1)
        else:
            sample_resuls.append(0)
    if sum(sample_resuls)> 0:
        reject_YN = 1
    else:
       reject_YN = 0

    if len(probs_detection)>1:
        pdetect = 1-np.prod([1-i for i in probs_detection])
    else: 
        pdetect = probs_detection[0]
    
    return [reject_YN,pdetect]

#%% Loading PCR Detection Models
filename_qPCR = 'C:\\Users\gareyes3\Documents\GitHub\CPS-Farm-To-Facility-Cilantro\logistic_AW_Testing_qPCR.sav'
qPCR_Model_AW = pickle.load(open(filename_qPCR, 'rb'))

filename_qPCR = 'C:\\Users\gareyes3\Documents\GitHub\CPS-Farm-To-Facility-Cilantro\logistic_Prod_Test_qPCR_FDA.sav'
qPCR_Model = pickle.load(open(filename_qPCR, 'rb'))
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

#3Product Sample
Sample_Weight = 25
N_25g_Samples = 1
N_Grabs = 1

Sampling_every_Days_Product = 3
Product_Sampling_Days = np.arange(1,Days_per_season,Sampling_every_Days_Product)

#%% output collection functions
def Output_DF_Creation(Column_Names, Niterations):
    Outputs_Df =pd.DataFrame(np.NaN, index= range(Niterations), columns =Column_Names)
    return Outputs_Df

def Output_Collection_ProduceCont(df, outputDF, Step_Column,iteration):
    #df= main model df
    #outputDF = contprogdataframe
    #Step_column = day of process
    outputDF.at[iteration,Step_Column] =np.array(df["Oo"]).sum()
    return outputDF

def Output_Collection_any_output(outputDF, Step_Column,iteration, outcome):
    #df= main model df
    #outputDF = contprogdataframe
    #Step_column = day of process
    outputDF.at[iteration,Step_Column] =outcome
    return outputDF

#%%

def Process_Model(Days_per_season,
                  Niterations,
                  Cont_Scenario, #1 = every day cont, 2 = one random day
                  Testing_Scenario,# 1=choose every so many days per seson, #2, choose when as input
                  #Contamination Information
                  OO_per_L,
                  #Water Testing Options
                  Sampling_every_Days_Water,
                  Sampling_every_Days_Product,
                  #for scenario 2
                  Testing_Day_Water,
                  Testing_Day_Product,
                  #Testing Options
                  Water_Sampling = 0,
                  Product_Sampling_PH = 0,
                  Product_Testing_H = 0
                  ):
    
    #Initial levels based on Input
    #Initial_Levels_Bulk = int(Total_L_Season*OO_per_L)
    #Irrigation_Levels_Days = Initial_Levels_Bulk/Days_per_season 
    
    #Dataframe output creation
    Water_Outcome_DF = Output_DF_Creation(Column_Names =np.arange(1,Days_per_season+1), Niterations= Niterations)
    Water_PrRej_DF = Output_DF_Creation(Column_Names =np.arange(1,Days_per_season+1), Niterations= Niterations)
    Produce_Outcome_DF = Output_DF_Creation(Column_Names =np.arange(1,Days_per_season+1), Niterations= Niterations)
    Produce_PrRej_DF = Output_DF_Creation(Column_Names =np.arange(1,Days_per_season+1), Niterations= Niterations)
    Contam_Produce_DF = Output_DF_Creation(Column_Names =np.arange(1,Days_per_season+1), Niterations= Niterations)
    Harvest_Sampling_Outcome_DF = Output_DF_Creation(Column_Names =[Days_per_season], Niterations= Niterations)
    Harvest_Sampling_PrRej_DF = Output_DF_Creation(Column_Names =[Days_per_season], Niterations= Niterations)
    Contam_HS_DF = Output_DF_Creation(Column_Names =[Days_per_season], Niterations= Niterations)
    Final_CFUS_DF = Output_DF_Creation(Column_Names =np.arange(1,Days_per_season+1), Niterations= Niterations)
    
    
    for k in (range(Niterations)): 
        print(k)
        
        if Cont_Scenario ==2:
            Random_Irr_Day_Scen2 = random.randint(1,Days_per_season)
                
        #Water Sampling
        #Sampling_every_Days_Water = 1
        if Testing_Scenario ==1:
            Water_Sampling_Days = np.arange(1,Days_per_season+1,Sampling_every_Days_Water)
        elif Testing_Scenario ==2:
            Water_Sampling_Days = [Testing_Day_Water]
        
        #Product Sampling Days
        #Sampling_every_Days_Product = 1
        if Testing_Scenario ==1:
            Product_Sampling_Days = np.arange(1,Days_per_season+1,Sampling_every_Days_Product)
        elif Testing_Scenario ==2:
            Product_Sampling_Days = [Testing_Day_Product]
        
        #Water_Sampling = 1
        #Product_Sampling_PH = 1
        #Product_Testing_H = 1
        
        
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
        
        W_Test_Outcome = 0
        
        for i in range (1,Days_per_season+1):
            
            #Changing water contmaination level for scenario 2: if not 2 then levels are same very day
            if Cont_Scenario ==2 and i != Random_Irr_Day_Scen2:
                Initial_Levels_Bulk = 0
                Irrigation_Levels_Days =0
            else:
                Initial_Levels_Bulk = int(Total_L_Season*OO_per_L)
                Irrigation_Levels_Days = Initial_Levels_Bulk/Days_per_season 
        
        
            #Water Sampling: Happens in sampling days
            if i in Water_Sampling_Days and Water_Sampling == 1 :
                W_Test_Outcome = Func_Water_Sampling (total_oocyst_bw = Initial_Levels_Bulk, 
                                                bw_volume =Total_L_Season , 
                                                sample_size_volume =W_Sample_Vol,
                                                total_samples = Total_Samples_Day, 
                                                loaded_model =qPCR_Model_AW )
                
                Water_Outcome_DF = Output_Collection_any_output(outputDF = Water_Outcome_DF, Step_Column = i,iteration=k, outcome = W_Test_Outcome[0])
                Water_PrRej_DF = Output_Collection_any_output(outputDF = Water_PrRej_DF, Step_Column = i,iteration=k, outcome = W_Test_Outcome[1])
                    
                
            #Irrigation with water
            #Step 1: Irrigation Event, only run if contamination is there: 
            if Irrigation_Levels_Days>0:
                Cilantro_df = field_cont_percetage2(df = Cilantro_df, 
                                                    percent_cont = Per_Cont_Field, 
                                                    Hazard_lvl =Irrigation_Levels_Days,
                                                    No_Cont_Clusters = 1)
                
            
            #Preharvest product testing. 
            if i in Product_Sampling_Days and Product_Sampling_PH == 1 :
                  Produce_test_results =Cilantro_Sampling_25g(df=Cilantro_df,
                                          Sample_Weight = Sample_Weight,
                                          N_25g_Samples = N_25g_Samples,
                                          N_Grabs_Sample = N_Grabs ,
                                          Plant_Weight = Plant_Weight, 
                                          loaded_model =  qPCR_Model)
              
                  Produce_Outcome_DF = Output_Collection_any_output(outputDF = Produce_Outcome_DF, Step_Column = i,iteration=k, outcome = Produce_test_results[1])
                  Produce_PrRej_DF = Output_Collection_any_output(outputDF = Produce_PrRej_DF, Step_Column = i,iteration=k, outcome = Produce_test_results[2])
                  Contam_Produce_DF=Output_Collection_ProduceCont(df=Cilantro_df, outputDF=Contam_Produce_DF, Step_Column = i,iteration = k)
        
        
            #Harvest Sampling
            if i == Days_per_season:
                #Harvest Testing
                if Product_Testing_H  == 1:
                    Harvest_Test_Results =Cilantro_Sampling_25g(df=Cilantro_df,
                                          Sample_Weight = Sample_Weight,
                                          N_25g_Samples = N_25g_Samples,
                                          N_Grabs_Sample = N_Grabs ,
                                          Plant_Weight = Plant_Weight, 
                                          loaded_model =  qPCR_Model)
                    
                    #Cilantro_df =F_Rejection_Rule_C (df= Cilantro_df )
                
                    Harvest_Sampling_Outcome_DF = Output_Collection_any_output(outputDF = Harvest_Sampling_Outcome_DF, Step_Column = i,iteration=k, outcome =  Harvest_Test_Results[1])
                    Harvest_Sampling_PrRej_DF = Output_Collection_any_output(outputDF = Harvest_Sampling_PrRej_DF, Step_Column = i,iteration=k, outcome =  Harvest_Test_Results[2])
                    Contam_HS_DF=Output_Collection_ProduceCont(df=Cilantro_df, outputDF=Contam_HS_DF, Step_Column = i,iteration = k)
            
            Final_CFUS_DF=Output_Collection_ProduceCont(df=Cilantro_df, outputDF=Final_CFUS_DF, Step_Column = i,iteration = k)

    return [Water_Outcome_DF,Water_PrRej_DF,Produce_Outcome_DF,Produce_PrRej_DF, Contam_Produce_DF,Harvest_Sampling_Outcome_DF,Harvest_Sampling_PrRej_DF,Contam_HS_DF,Final_CFUS_DF]


    
        

        
#Baseline Scenario 1, No Sampling one a day. Contamination every Day
#Low Contamination
Baseline_1_L = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10,
                  Cont_Scenario = 1,
                  Testing_Scenario=2,
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 0,
                  Sampling_every_Days_Product = 0,
                  #Testing Options
                  Testing_Day_Water = 0,
                  Testing_Day_Product = 0,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 0,
                  Product_Testing_H = 0
                  )

#High Contamination
Baseline_1_H = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10,
                  Cont_Scenario = 1,
                  Testing_Scenario=2,
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 0,
                  Sampling_every_Days_Product = 0,
                  #Testing Options
                  Testing_Day_Water = 0,
                  Testing_Day_Product = 0,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 0,
                  Product_Testing_H = 0
                  )

Baseline_1_L = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10,
                  Cont_Scenario = 1,
                  Testing_Scenario=2,
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 0,
                  Sampling_every_Days_Product = 0,
                  #Testing Options
                  Testing_Day_Water = 0,
                  Testing_Day_Product = 0,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 0,
                  Product_Testing_H = 0
                  )




#Baseline Scenario 1, No Sampling one a day
Baseline_2_L = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10,
                  Cont_Scenario = 1,
                  Testing_Scenario=1,
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1,
                  Sampling_every_Days_Product = 1,
                  #Testing Options
                  Testing_Day_Water = 0,
                  Testing_Day_Product = 0,
                  Water_Sampling = 1,
                  Product_Sampling_PH = 0,
                  Product_Testing_H = 0
                  )
#%%
Iterations = 10

df_ext = pd.DataFrame({"Iteration": range(Iterations),
                   "MinDay": ""})
for i in range(Iterations):
    list_of_days=list(range(1,45+1))
    index_rows =Baseline_2_L[0].iloc[[i]] ==1
    index_rows=index_rows.values.flatten().tolist()
    postivie_days=[i+1 for i, x in enumerate(index_rows) if x]
    if len(postivie_days)>0:
        min_day= min(postivie_days)
    else:
        min_day = np.nan
    df_ext.loc[i,"MinDay"] =  min_day

import seaborn as sns

Baseline_1[1].melt()   
        
        
sns.lineplot(data= Baseline_1[3].melt() , x = "variable", y = "value")    
        
    
#Best Day product testing
Days_Positive = Baseline_1[2].melt()
Days_Positive = Days_Positive[Days_Positive["value"] == 1]
sns.histplot(data = Days_Positive, x = "variable",bins = 45)
    
    
#Best Day water testing
Days_Positive = Baseline_1[0].melt()
Days_Positive = Days_Positive[Days_Positive["value"] == 1]
sns.histplot(data = Days_Positive, x = "variable", bins = 45) 
    

