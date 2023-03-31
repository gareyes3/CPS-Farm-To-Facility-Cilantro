# -*- coding: utf-8 -*-

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
    while choice == nb_chunks-1 and nb_chunks!= 1:
       choice = random.randrange(nb_chunks) 
    return lst[choice*chunk_size:(choice+1)*chunk_size]

def field_cont_percetage2(df, percent_cont, Hazard_lvl,No_Cont_Clusters):
    df2=df.copy()
    No_Cont_Clusters = 1
    #This function contaminated the tomato field randomly based on a cluter of X%. 
    No_Cont_PartitionUnits= int(len(df2)*(percent_cont/100))
    Field_df_1 =df2.loc[(df2["Location"]==1) & (df2["Rej_Acc"]=="Acc")].copy()
    if len(Field_df_1)>0:
        Hazard_lvl_percluster= Hazard_lvl /No_Cont_Clusters #(No_Cont_PartitionUnits*No_Cont_Clusters)
        for i in range(0,No_Cont_Clusters):
            random_Chunky = np.array(random_chunk(lst = df2.index, chunk_size = No_Cont_PartitionUnits)) #creating random chunk
            Contamination_Pattern = np.random.multinomial(Hazard_lvl_percluster,[1/No_Cont_PartitionUnits]*No_Cont_PartitionUnits,1) #spliting cont into chunks length
            #print(len(Contamination_Pattern[0]))
            random_Chunky_s= random_Chunky[np.isin(random_Chunky,np.array(Field_df_1.index))]
            #print(len(Field_df_1.index))
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
    if Total_Oo_Composite>=1:
        Pr_Detect = loaded_model.predict_proba(np.array([Total_Oo_Composite]).reshape(-1,1))[0][1]
    else:
        Pr_Detect = 0
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
        if Total_Oo_Composite>=1:
            Pr_Detect = loaded_model.predict_proba(np.array([Total_Oo_Composite]).reshape(-1,1))[0][1] #from logistic
        else:
            Pr_Detect = 0       
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
        if T_Ocyst_SBW>=1:
            Pr_Detect = loaded_model.predict_proba(np.array([T_Ocyst_SBW]).reshape(-1,1))[0][1] #from logistic
        else: 
            Pr_Detect = 0 
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
filename_qPCR = 'C://Users/gareyes3/Documents/GitHub/CPS-Farm-To-Facility-Cilantro/logistic_AW_Testing_qPCR.sav'
qPCR_Model_AW = pickle.load(open(filename_qPCR, 'rb'))
qPCR_Model_AW.predict_proba(np.array([20]).reshape(-1,1))[0][1] #from logistic

filename_qPCR = 'C://Users/gareyes3/Documents/GitHub/CPS-Farm-To-Facility-Cilantro/logistic_Prod_Test_qPCR_FDA.sav'
qPCR_Model = pickle.load(open(filename_qPCR, 'rb'))

#qPCR_Model.predict_proba(np.array([20]).reshape(-1,1))[0][1] #from logistic
#%%

#Cilantro Static Inputs

#Creating the Field
Field_Yield = 22_000 #lb
Plant_Weight = 1 #lb
Total_Plants = int(Field_Yield/Plant_Weight)
Total_Plants_List = range(1,Total_Plants+1)

#not in use because until harvest
Case_Weight = 20 #lb per case
Bunches_Weight = Case_Weight/Plant_Weight

#Water and Season Characteristics
Water_Irrigation_In = 12 #Inches of water per harvest season
Total_L_Season = 40.46*40.46*(0.0254*Water_Irrigation_In)*1000 # one acre 40.46m2 * 0.348 m of water * 10000 to convert tot m3
Days_per_season = 45 #days
L_water_day = Total_L_Season/Days_per_season

#Contamination
#percentage of field contaminated
#Per_Cont_Field = 100

#Water testing characteristics
W_Sample_Vol = 10 #L
Total_Samples_Water = 1

#Product Sample
Sample_Weight = 25 #g
N_25g_Samples = 1
N_Grabs = 1


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

#%% Function to run the scenarions in the process mdoel

def Process_Model(Days_per_season,
                  Niterations,
                  Cont_Scenario, #1 = every day cont, 2 = one random day
                  Testing_Scenario,# 1=choose every so many days per seson, #2, choose when as input
                  #Contamination Information, you can also use this for different clustering.
                  OO_per_L,
                  #Water Testing Options
                  Sampling_every_Days_Water, #change to 1 always
                  Sampling_every_Days_Product, #change to 1 always
                  #for scenario 2
                  Testing_Day_Water, #if scenario 2, then select the days you want testing to happen 1,23,45
                  Testing_Day_Product,#if scenario 2, then select the days you want testing to happen 1,23,45
                  #Field Contamination
                  Per_Cont_Field = 100,
                  #Testing Options
                  Water_Sampling = 0,#1 is on 0 is off
                  Product_Sampling_PH = 0, #1 is on 0 is off
                  Product_Testing_H = 0 ##1 is on 0 is off
                  ):
    
    #Initial levels based on Input
    #Initial_Levels_Bulk = int(Total_L_Season*OO_per_L)
    #Irrigation_Levels_Days = Initial_Levels_Bulk/Days_per_season 
    
    #Dataframe output creation, THIS ARE THE [0], [2], [8] later in the output section
    
    #tells you  what if water rejected on a given day
    Water_Outcome_DF = Output_DF_Creation(Column_Names =np.arange(1,Days_per_season+1), Niterations= Niterations)
    #tells you the probability of detection per days
    Water_PrRej_DF = Output_DF_Creation(Column_Names =np.arange(1,Days_per_season+1), Niterations= Niterations)
    #tells you if produce sampling detected every day
    Produce_Outcome_DF = Output_DF_Creation(Column_Names =np.arange(1,Days_per_season+1), Niterations= Niterations)
    #tell you probability of detection of product sampling
    Produce_PrRej_DF = Output_DF_Creation(Column_Names =np.arange(1,Days_per_season+1), Niterations= Niterations)
    #tells you contamination at each sampling point per day
    Contam_Produce_DF = Output_DF_Creation(Column_Names =np.arange(1,Days_per_season+1), Niterations= Niterations)
    #not in use harest samplfin
    Harvest_Sampling_Outcome_DF = Output_DF_Creation(Column_Names =[Days_per_season], Niterations= Niterations)
    Harvest_Sampling_PrRej_DF = Output_DF_Creation(Column_Names =[Days_per_season], Niterations= Niterations)
    Contam_HS_DF = Output_DF_Creation(Column_Names =[Days_per_season], Niterations= Niterations)
    #Tells you the cells at the end of each day
    Final_CFUS_DF = Output_DF_Creation(Column_Names =np.arange(1,Days_per_season+1), Niterations= Niterations)
    
    
    for k in (range(Niterations)): 
        print(k)
        
        Random_Irr_Day_Scen2 = 1
        #Contmaination scenario selection
        if Cont_Scenario ==2:
            Random_Irr_Day_Scen2 = random.randint(1,Days_per_season)
                
        #Water Sampling
        #Sampling_every_Days_Water = 1
        if Testing_Scenario ==1 :
            Water_Sampling_Days = np.arange(1,Days_per_season+1,Sampling_every_Days_Water)
        elif Testing_Scenario ==2:
            Water_Sampling_Days = Testing_Day_Water
        
        #Product Sampling Days
        #Sampling_every_Days_Product = 1
        if Testing_Scenario ==1 :
            Product_Sampling_Days = np.arange(1,Days_per_season+1,Sampling_every_Days_Product)
        elif Testing_Scenario ==2:
            Product_Sampling_Days =Testing_Day_Product
        
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
            if Cont_Scenario ==2 and i == Random_Irr_Day_Scen2:
                Initial_Levels_Bulk = int(Total_L_Season*OO_per_L)
                Irrigation_Levels_Days = Initial_Levels_Bulk/Days_per_season 
            elif Cont_Scenario ==1:
                Initial_Levels_Bulk = int(Total_L_Season*OO_per_L)
                Irrigation_Levels_Days = Initial_Levels_Bulk/Days_per_season 
        
        
            #Water Sampling: Happens in sampling days
            if i in Water_Sampling_Days and Water_Sampling == 1 :
                W_Test_Outcome = Func_Water_Sampling (total_oocyst_bw = Initial_Levels_Bulk, 
                                                bw_volume =Total_L_Season , 
                                                sample_size_volume =W_Sample_Vol,
                                                total_samples = Total_Samples_Water, 
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

#a_Tring = Process_Model(
 #                 Days_per_season = 45,
 #                 Niterations= 10,
 #                 Cont_Scenario = 2,
 #                 Testing_Scenario=2,#Testing only in given day
 #                 #Contamination Information
 #                 OO_per_L =0.06,
 #                 #Water Testing Options
 #                 Sampling_every_Days_Water = 1, #leave as 1 defaul
 #                 Sampling_every_Days_Product = 1, #as default 
 #                 #Testing Options
 #                 Testing_Day_Water = [1], #testing water on day 1
 #                 Testing_Day_Product = [0],
 #                 Water_Sampling = 1,#now water testing is on
 #                 Product_Sampling_PH = 0, 
 #                 Product_Testing_H = 0
#                  )

    
 #%% SCENARIOS     THESE ARE EXAMPLES (SOME REAL SCENARIO), TUNE THEM TO MATCH THE SCENARIOS WE SPOKE ABOUT  IN DOC
#FOR THE FINAL ANALYSIS nITERATIONS ==10,000  
       
#Baseline Scenario 1, No Sampling one a day. Contamination every Day
#Low Contamination
Out_B1_L = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,
                  Testing_Scenario=2,
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 0,
                  Sampling_every_Days_Product = 0,
                  #Testing Options
                  Testing_Day_Water = [0],
                  Testing_Day_Product = [0],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 0,
                  Product_Testing_H = 0
                  )

#High Contamination
Out_B1_H = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,#Change in baseline scenario
                  Testing_Scenario=2,#leave as 2 default
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 0,
                  Sampling_every_Days_Product = 0,
                  #Testing Options
                  Testing_Day_Water = [0],
                  Testing_Day_Product = [0],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 0,
                  Product_Testing_H = 0
                  )

#Baseline Scenario 2: Irrigation one day randomly
Out_B2_L = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2, #Change in baseline scenario
                  Testing_Scenario=2, #leave as 2 default
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 0,
                  Sampling_every_Days_Product = 0,
                  #Testing Options
                  Testing_Day_Water = [0],
                  Testing_Day_Product = [0],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 0,
                  Product_Testing_H = 0
                  )

Out_B2_H = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2, #Change in baseline scenario
                  Testing_Scenario=2, #leave as 2 default
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 0,
                  Sampling_every_Days_Product = 0,
                  #Testing Options
                  Testing_Day_Water = [0],
                  Testing_Day_Product = [0],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 0,
                  Product_Testing_H = 0
                  )



#BaselineScenario 1: Plus sampling plans =========================================

#B1_Daily testing water (DTW) - Low
Scen_B1_L_DTW = Process_Model(
                  Days_per_season = 45,
                  Niterations= 100,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=1,#every day sampling
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0],
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,
                  Product_Sampling_PH = 0,
                  Product_Testing_H = 0
                  )

Scen_B1_L_DTW[0].mean(axis=1)

#B1_Daily testing water (DTW) - High
Scen_B1_H_DTW = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=1,#every day sampling
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0],
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,
                  Product_Sampling_PH = 0,
                  Product_Testing_H = 0
                  )


#B1_Daily testing product (DTP) - Low
Scen_B1_L_DTP = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=1,#every day sampling
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [0],
                  Testing_Day_Product = [0],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, #now product testing is on
                  Product_Testing_H = 0
                  )
#B1_Daily testing product (DTP) - High
Scen_B1_H_DTP = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=1,#every day sampling
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [0],
                  Testing_Day_Product = [0],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, #now product testing is on
                  Product_Testing_H = 0
                  )

#B1_Water testing 1 time per season at the start of the season - Low
Scen_B1_L_WT1 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1], #testing water on day 1
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 0, 
                  Product_Testing_H = 0
                  )

#B1_Water testing 1 time per season at the start of the season - High
Scen_B1_H_WT1 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1], #testing water on day 1
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 0, 
                  Product_Testing_H = 0
                  )

#B1_Water testing 2 times per season (1 start, 1 end of the season) - Low
Scen_B1_L_WT2 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1,45], #testing water on day 1
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 0, 
                  Product_Testing_H = 0
                  )
#B1_Water testing 2 times per season (1 start, 1 end of the season) - High
Scen_B1_H_WT2 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1,45], #testing water on day 1
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 0, 
                  Product_Testing_H = 0
                  )

#B1_Water testing 3 times per season (1 start, 1 end of the season) - Low
Scen_B1_L_WT3 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1,22,45], #testing water on day 1
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 0, 
                  Product_Testing_H = 0
                  )

#B1_Water testing 3 times per season (1 start, 1 end of the season) - High
Scen_B1_H_WT3 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1,22,45], #testing water on day 1
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 0, 
                  Product_Testing_H = 0
                  )

#B1_product testing 1 time per season at the start of the season - Low
Scen_B1_L_PT1 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [0], #testing water on day 1
                  Testing_Day_Product = [1],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

#B1_product testing 1 time per season at the start of the season - High
Scen_B1_H_PT1 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 100,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [1],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )



#B1_product testing 2 times per season (1 start, 1 end of the season) - Low
Scen_B1_L_PT2 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [1,45],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )
#B1_product testing 2 times per season (1 start, 1 end of the season) - High
Scen_B1_H_PT2 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 100,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [1,45],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

#B1_product testing 3 times per season (1 start, 1 end of the season) - Low
Scen_B1_L_PT3 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [0], #testing water on day 1
                  Testing_Day_Product = [1,22,45],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

#B1_product testing 3 times per season (1 start, 1 end of the season) - High
Scen_B1_H_PT3 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1,  
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [1,22,45],#testing water on day 1
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, #now product testing is on
                  Product_Testing_H = 0
                  )

#BaselineScenario 2: Plus sampling plans =========================================

#B2_Daily testing water (DTW) - Low
Scen_B2_L_DTW = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,
                  Testing_Scenario=1,#every day sampling
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0],
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,
                  Product_Sampling_PH = 0,
                  Product_Testing_H = 0
                  )
#B2_Daily testing water (DTW) - High
Scen_B2_H_DTW = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,
                  Testing_Scenario=1,#every day sampling
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0],
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,
                  Product_Sampling_PH = 0,
                  Product_Testing_H = 0
                  )


#B2_Daily testing product (DTP) - Low
Scen_B2_L_DTP = Process_Model(
                  Days_per_season = 45,
                  Niterations= 100,
                  Cont_Scenario = 2,
                  Testing_Scenario=1,#every day sampling
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [0],
                  Testing_Day_Product = [0],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, #now product testing is on
                  Product_Testing_H = 0
                  )


#B2_Daily testing product (DTP) - High
Scen_B2_H_DTP = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,
                  Testing_Scenario=1,#every day sampling
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [0],
                  Testing_Day_Product = [0],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, #now product testing is on
                  Product_Testing_H = 0
                  )

#B2_Water testing 1 time per season at the start of the season - Low
Scen_B2_L_WT1 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1], #testing water on day 1
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 0, 
                  Product_Testing_H = 0
                  )

#B2_Water testing 1 time per season at the start of the season - High
Scen_B2_H_WT1 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1], #testing water on day 1
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 0, 
                  Product_Testing_H = 0
                  )

#B2_Water testing 2 times per season (1 start, 1 end of the season) - Low
Scen_B2_L_WT2 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1,45], #testing water on day 1
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 0, 
                  Product_Testing_H = 0
                  )
#B2_Water testing 2 times per season (1 start, 1 end of the season) - High
Scen_B2_H_WT2 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1,45], #testing water on day 1
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 0, 
                  Product_Testing_H = 0
                  )
###
#B2_Water testing 3 times per season (1 start, 1 end of the season) - Low
Scen_B2_L_WT3 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1,22,45], #testing water on day 1
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 0, 
                  Product_Testing_H = 0
                  )

#B2_Water testing 3 times per season (1 start, 1 end of the season) - High
Scen_B2_H_WT3 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1,22,45], #testing water on day 1
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 0, 
                  Product_Testing_H = 0
                  )

#B2_product testing 1 time per season at the start of the season - Low
Scen_B2_L_PT1 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [0], #testing water on day 1
                  Testing_Day_Product = [1],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

#B2_product testing 1 time per season at the start of the season - High
Scen_B2_H_PT1 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [1],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

#B2_product testing 2 times per season (1 start, 1 end of the season) - Low
Scen_B2_L_PT2 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [1,45],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )
#B2_product testing 2 times per season (1 start, 1 end of the season) - High
Scen_B2_H_PT2 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [1,45],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

#B2_product testing 3 times per season (1 start, 1 end of the season) - Low
Scen_B2_L_PT3 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [0], #testing water on day 1
                  Testing_Day_Product = [1,22,45],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

#B2_product testing 3 times per season (1 start, 1 end of the season) - High
Scen_B2_H_PT3 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1,  
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [1,22,45],#testing water on day 1
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, #now product testing is on
                  Product_Testing_H = 0
                  )
# Water testing with product testing Baseline 1 ==============

Scen_B1_H_DWPT = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,
                  Testing_Scenario=1,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1,  
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Water_Sampling = 1,
                  Product_Sampling_PH = 1, #now product testing is on
                  Product_Testing_H = 0
                  )

Scen_B1_L_DWPT = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,
                  Testing_Scenario=1,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1,  
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Water_Sampling = 1,
                  Product_Sampling_PH = 1, #now product testing is on
                  Product_Testing_H = 0
                  )


#B1_Daily testing product (DTP) - High
Scen_B1_H_WPT1 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=2,#every day sampling
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1],
                  Testing_Day_Product = [1],
                  Water_Sampling = 1,
                  Product_Sampling_PH = 1, #now product testing is on
                  Product_Testing_H = 0
                  )

#B1_Water testing 1 time per season at the start of the season - Low
Scen_B1_L_WPT1 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1], #testing water on day 1
                  Testing_Day_Product = [1],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

#B1_Water testing 1 time per season at the start of the season - High
Scen_B1_H_WPT2 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1,45], #testing water on day 1
                  Testing_Day_Product = [1,45],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

#B1_Water testing 2 times per season (1 start, 1 end of the season) - Low
Scen_B1_L_WPT2 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1,45], #testing water on day 1
                  Testing_Day_Product = [1,45],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )
#B1_Water testing 2 times per season (1 start, 1 end of the season) - High
Scen_B1_H_WPT3 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1,22,45], #testing water on day 1
                  Testing_Day_Product = [1,22,45],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

#B1_Water testing 3 times per season (1 start, 1 end of the season) - Low
Scen_B1_L_WPT3 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1,22,45], #testing water on day 1
                  Testing_Day_Product = [1,22,45],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

# Water testing with product testing Baseline 2 ==============

Scen_B2_H_DWPT = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,
                  Testing_Scenario=1,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1,  
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Water_Sampling = 1,
                  Product_Sampling_PH = 1, #now product testing is on
                  Product_Testing_H = 0
                  )

Scen_B2_L_DWPT = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,
                  Testing_Scenario=1,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1,  
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Water_Sampling = 1,
                  Product_Sampling_PH = 1, #now product testing is on
                  Product_Testing_H = 0
                  )


#B1_Daily testing product (DTP) - High
Scen_B2_H_WPT1 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,#every day contamiantion
                  Testing_Scenario=2,#every day sampling
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1],
                  Testing_Day_Product = [1],
                  Water_Sampling = 1,
                  Product_Sampling_PH = 1, #now product testing is on
                  Product_Testing_H = 0
                  )

#B1_Water testing 1 time per season at the start of the season - Low
Scen_B2_L_WPT1 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1], #testing water on day 1
                  Testing_Day_Product = [1],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

#B1_Water testing 1 time per season at the start of the season - High
Scen_B2_H_WPT2 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1,45], #testing water on day 1
                  Testing_Day_Product = [1,45],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

#B1_Water testing 2 times per season (1 start, 1 end of the season) - Low
Scen_B2_L_WPT2 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1,45], #testing water on day 1
                  Testing_Day_Product = [1,45],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )
#B1_Water testing 2 times per season (1 start, 1 end of the season) - High
Scen_B2_H_WPT3 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1,22,45], #testing water on day 1
                  Testing_Day_Product = [1,22,45],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

#B1_Water testing 3 times per season (1 start, 1 end of the season) - Low
Scen_B2_L_WPT3 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [1,22,45], #testing water on day 1
                  Testing_Day_Product = [1,22,45],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )



#%% Generating function for outputs

#Output 1: TAC at system endpoint. If rejected at any point Endpoint TACs = 0 

#Extracting Rejection Days: 
    
#gets the earliest day of each iteration when the product was rejected.
def Rejection_Days(Iterations, dfrejections):    
    df_ext = pd.DataFrame({"Iteration": range(Iterations),
                       "MinDay": ""})
    for i in range(Iterations):
        index_rows =dfrejections.iloc[[i]] ==1
        index_rows=index_rows.values.flatten().tolist()
        postivie_days=[i+1 for i, x in enumerate(index_rows) if x]
        if len(postivie_days)>0:
            min_day= min(postivie_days)
        else:
            min_day = "Never Rejected"
        df_ext.loc[i,"MinDay"] =  min_day
    return  df_ext
        
#gets the endpoint TAC of any days that were not able to reject contamination
def get_exposure(df_rejections,df_FinalConts,Iterations, Days_per_season):
    Days_Rejected=Rejection_Days(Iterations = Iterations, dfrejections =df_rejections )
    
    #Checking for NA Values.
    Not_Rejected = Days_Rejected["MinDay"] == "Never Rejected"
    Index_Not_Rejected=[i for i, x in enumerate(Not_Rejected) if x==True]
    exps =df_FinalConts[Days_per_season][Index_Not_Rejected].sum()
    return exps

def get_pdetect(df, Water_Produce_Both):
    if Water_Produce_Both == "Water":
        data = df[1]
    elif Water_Produce_Both == "Produce":
        data = df[3]
        
    Pdetects = []
    for i in list(range(len(data))):
        Probs = data.loc[[i]].values.tolist()[0]
        Probs_g_zero = [i for i in Probs if i >0]
        if len(Probs_g_zero)>0:
            Probs_g_zero_rem = [1-i for i in Probs_g_zero]
            Prod_Probs_g_zero_rem=np.prod(Probs_g_zero_rem)
            Pdetect = 1-Prod_Probs_g_zero_rem
        else:
            Pdetect = 0 
        Pdetects.append(Pdetect)
    return Pdetects

def get_dec_rate(df, Water_Produce_Both):
    if Water_Produce_Both == "Water":
        data = df[0]
    elif Water_Produce_Both == "Produce":
        data = df[2]
    return [np.mean(data.sum(axis = 1)>0)]
    

def get_exposure_WP(df_rejections_W,df_rejections_P,df_FinalConts,Iterations, Days_per_season):
    Days_Rejected_W=Rejection_Days(Iterations = Iterations, dfrejections =df_rejections_W )
    Days_Rejected_P=Rejection_Days(Iterations = Iterations, dfrejections =df_rejections_P )

    #Checking for NA Values.
    Not_Rejected_W = Days_Rejected_W["MinDay"] == "Never Rejected"
    Not_Rejected_P = Days_Rejected_P["MinDay"] == "Never Rejected"

 

    
    Index_Not_Rejected_W=[i for i, x in enumerate(Not_Rejected_W) if x==True]
    Index_Not_Rejected_P=[i for i, x in enumerate(Not_Rejected_P) if x==True]

    exps_W =df_FinalConts[Days_per_season][Index_Not_Rejected_W].sum()
    exps_P =df_FinalConts[Days_per_season][Index_Not_Rejected_P].sum()

    exps = min(exps_W, exps_P)


    return exps

#%%

Scenario_Names=["Daily_Water_Test",
"Daily_Produce_Test",
"Water_Testing_1",
"Water_Testing_2",
"Water_Testing_3",
"Produce_Testing_1",
"Produce_Testing_2",
'Produce_Testing_3',
"Daily_Water&Product_Test",
"Water&Product_1",
"Water&Product_2",
"Water&Product_3"]

Baseline_1_df_H = pd.DataFrame({"Scenario":Scenario_Names,
                                "Exp" :""})
Baseline_1_df_L = pd.DataFrame({"Scenario":Scenario_Names,
                                "Exp" :""})

#THIS GIVES YOU THE ENDPOINT TAC, ADJUST ACCORDINGLY PER SCENARIO

get_exposure(df_rejections =Out_B1_L[0] ,df_FinalConts = Out_B1_L[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Out_B1_H[0] ,df_FinalConts = Out_B1_H[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Out_B2_L[0] ,df_FinalConts = Out_B2_L[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Out_B2_H[0] ,df_FinalConts = Out_B2_H[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

#BaselineScenario 1: Plus sampling plans =========================================

get_exposure(df_rejections =Scen_B1_H_DTP[2] ,df_FinalConts = Scen_B1_H_DTP[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B1_H_DTW[0] ,df_FinalConts = Scen_B1_H_DTW[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B1_H_PT1[2] ,df_FinalConts = Scen_B1_H_PT1[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B1_H_PT2[2] ,df_FinalConts = Scen_B1_H_PT2[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B1_H_PT3[2] ,df_FinalConts = Scen_B1_H_PT3[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B1_H_WT1[0] ,df_FinalConts = Scen_B1_H_WT1[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B1_H_WT2[0] ,df_FinalConts = Scen_B1_H_WT2[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B1_H_WT3[0] ,df_FinalConts = Scen_B1_H_WT3[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing


get_exposure_WP(df_rejections_W = Scen_B1_H_DWPT[0],df_rejections_P = Scen_B1_H_DWPT[2],df_FinalConts = Scen_B1_H_DWPT[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure_WP(df_rejections_W = Scen_B1_H_WPT1[0],df_rejections_P = Scen_B1_H_WPT1[2],df_FinalConts = Scen_B1_H_WPT1[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure_WP(df_rejections_W = Scen_B1_H_WPT2[0],df_rejections_P = Scen_B1_H_WPT2[2],df_FinalConts = Scen_B1_H_WPT2[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure_WP(df_rejections_W = Scen_B1_H_WPT3[0],df_rejections_P = Scen_B1_H_WPT3[2],df_FinalConts = Scen_B1_H_WPT3[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing





get_exposure(df_rejections =Scen_B1_L_DTP[2] ,df_FinalConts = Scen_B1_L_DTP[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B1_L_DTW[0] ,df_FinalConts = Scen_B1_L_DTW[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B1_L_PT1[2] ,df_FinalConts = Scen_B1_L_PT1[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B1_L_PT2[2] ,df_FinalConts = Scen_B1_L_PT2[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B1_L_PT3[2] ,df_FinalConts = Scen_B1_L_PT3[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B1_L_WT1[0] ,df_FinalConts = Scen_B1_L_WT1[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B1_L_WT2[0] ,df_FinalConts = Scen_B1_L_WT2[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B1_L_WT3[0] ,df_FinalConts = Scen_B1_L_WT3[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing


get_exposure_WP(df_rejections_W = Scen_B1_L_DWPT[0],df_rejections_P = Scen_B1_L_DWPT[2],df_FinalConts = Scen_B1_L_DWPT[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure_WP(df_rejections_W = Scen_B1_L_WPT1[0],df_rejections_P = Scen_B1_L_WPT1[2],df_FinalConts = Scen_B1_L_WPT1[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure_WP(df_rejections_W = Scen_B1_L_WPT2[0],df_rejections_P = Scen_B1_L_WPT2[2],df_FinalConts = Scen_B1_L_WPT2[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure_WP(df_rejections_W = Scen_B1_L_WPT3[0],df_rejections_P = Scen_B1_L_WPT3[2],df_FinalConts = Scen_B1_L_WPT3[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing



#BaselineScenario 2: Plus sampling plans =========================================

get_exposure(df_rejections =Scen_B2_H_DTP[2] ,df_FinalConts = Scen_B2_H_DTP[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B2_H_DTW[0] ,df_FinalConts = Scen_B2_H_DTW[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B2_H_PT1[2] ,df_FinalConts = Scen_B2_H_PT1[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B2_H_PT2[2] ,df_FinalConts = Scen_B2_H_PT2[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B2_H_PT3[2] ,df_FinalConts = Scen_B2_H_PT3[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B2_H_WT1[0] ,df_FinalConts = Scen_B2_H_WT1[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B2_H_WT2[0] ,df_FinalConts = Scen_B2_H_WT2[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B2_H_WT3[0] ,df_FinalConts = Scen_B2_H_WT3[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing


get_exposure_WP(df_rejections_W = Scen_B2_H_DWPT[0],df_rejections_P = Scen_B2_H_DWPT[2],df_FinalConts = Scen_B2_H_DWPT[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure_WP(df_rejections_W = Scen_B2_H_WPT1[0],df_rejections_P = Scen_B2_H_WPT1[2],df_FinalConts = Scen_B2_H_WPT1[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure_WP(df_rejections_W = Scen_B2_H_WPT2[0],df_rejections_P = Scen_B2_H_WPT2[2],df_FinalConts = Scen_B2_H_WPT2[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure_WP(df_rejections_W = Scen_B2_H_WPT3[0],df_rejections_P = Scen_B2_H_WPT3[2],df_FinalConts = Scen_B2_H_WPT3[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing





get_exposure(df_rejections =Scen_B2_L_DTP[2] ,df_FinalConts = Scen_B2_L_DTP[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B2_L_DTW[0] ,df_FinalConts = Scen_B2_L_DTW[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B2_L_PT1[2] ,df_FinalConts = Scen_B2_L_PT1[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B2_L_PT2[2] ,df_FinalConts = Scen_B2_L_PT2[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B2_L_PT3[2] ,df_FinalConts = Scen_B2_L_PT3[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B2_L_WT1[0] ,df_FinalConts = Scen_B2_L_WT1[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B2_L_WT2[0] ,df_FinalConts = Scen_B2_L_WT2[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure(df_rejections =Scen_B2_L_WT3[0] ,df_FinalConts = Scen_B2_L_WT3[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing


get_exposure_WP(df_rejections_W = Scen_B2_L_DWPT[0],df_rejections_P = Scen_B2_L_DWPT[2],df_FinalConts = Scen_B2_L_DWPT[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure_WP(df_rejections_W = Scen_B2_L_WPT1[0],df_rejections_P = Scen_B2_L_WPT1[2],df_FinalConts = Scen_B2_L_WPT1[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure_WP(df_rejections_W = Scen_B2_L_WPT2[0],df_rejections_P = Scen_B2_L_WPT2[2],df_FinalConts = Scen_B2_L_WPT2[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing

get_exposure_WP(df_rejections_W = Scen_B2_L_WPT3[0],df_rejections_P = Scen_B2_L_WPT3[2],df_FinalConts = Scen_B2_L_WPT3[8],Iterations = 10000,Days_per_season=45) #0 is for  water testing




#get_exposure_WP(df_rejections_W = Scen_B1_H_DWPT[0],df_rejections_P = Scen_B1_H_DWPT[2],df_FinalConts = Scen_B1_H_DWPT[8],Iterations =10, Days_per_season =45)

#This should get the exposure of any of the scenarios: just replace the dataframe. at the end this can be automated. as well
#get_exposure(df_rejections =Scen_B1_L_DTW[0] ,df_FinalConts = Scen_B1_L_DTW[8],Iterations = 10,Days_per_season=45) #0 is for  water testing

#this gets you the output TAC  of scenario with product ttesting
#get_exposure(df_rejections =Scen_B1_L_DTWP[2] ,df_FinalConts = Scen_B1_L_DTWP[8],Iterations = 10, Days_per_season=45) #2 is for product testign product ttesting

#enpoint TAC for tttesting only on day 1 water
#get_exposure(df_rejections =Scen_B1_L_WT1[0] ,df_FinalConts = Scen_B1_L_WT1[8],Iterations = 10, Days_per_season=45) #2 is for  product testing

#%%

##NEW metric, getting Prob Accept

get_pdetect(df= Scen_B1_H_PT1, Water_Produce_Both= "Produce")


#%% Field Scenarios
#100% Cluster
 ##LOW 
#Product Testing every day 

Scen_100_L_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 100,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

Scen_50_L_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 50,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

Scen_25_L_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 25,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

Scen_12_L_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 12.5,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

Scen_6_L_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 6.25,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

Scen_3_L_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 3.125,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

Scen_1_L_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 1.5,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

Scen_075_L_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 0.75,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

Scen_04_L_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 0.4,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )


Scen_02_L_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 0.2,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

col_name_prod=["100%",
               "50%",
               "25%",
               "12.5%",
               "6.2%",
               "3.1%",
               "1.5%",
               "0.75%",
               "0.4%",
               "0.2%"
               ]

df_prod_Low = pd.DataFrame(columns=[col_name_prod])

df_prod_Low["100%"] = get_dec_rate(df= Scen_100_L_PTD , Water_Produce_Both= "Produce")
df_prod_Low["50%"] = get_dec_rate(df= Scen_50_L_PTD , Water_Produce_Both= "Produce")
df_prod_Low["25%"] = get_dec_rate(df= Scen_25_L_PTD , Water_Produce_Both= "Produce")
df_prod_Low["12.5%"] = get_dec_rate(df= Scen_12_L_PTD , Water_Produce_Both= "Produce")
df_prod_Low["6.2%"] = get_dec_rate(df= Scen_6_L_PTD , Water_Produce_Both= "Produce")
df_prod_Low["3.1%"] = get_dec_rate(df= Scen_3_L_PTD , Water_Produce_Both= "Produce")
df_prod_Low["1.5%"] = get_dec_rate(df= Scen_1_L_PTD , Water_Produce_Both= "Produce")
df_prod_Low["0.75%"] = get_dec_rate(df= Scen_075_L_PTD , Water_Produce_Both= "Produce")
df_prod_Low["0.4%"] = get_dec_rate(df= Scen_04_L_PTD , Water_Produce_Both= "Produce")
df_prod_Low["0.2%"] = get_dec_rate(df= Scen_02_L_PTD , Water_Produce_Both= "Produce")



##High contamiantion


Scen_100_H_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 100,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

Scen_50_H_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 50,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

Scen_25_H_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 25,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

Scen_12_H_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 12.5,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

Scen_6_H_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 6.25,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

Scen_3_H_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 3.125,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

Scen_1_H_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 1.5,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

Scen_075_H_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 0.75,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

Scen_04_H_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 0.4,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )


Scen_02_H_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 0.2,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

Scen_01_H_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 0.1,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

col_name_prod=["100%",
               "50%",
               "25%",
               "12.5%",
               "6.2%",
               "3.1%",
               "1.5%",
               "0.75%",
               "0.4%",
               "0.2%",
               "0.1%"
               ]

df_prod_High = pd.DataFrame(columns=[col_name_prod])

df_prod_High["100%"] = get_dec_rate(df= Scen_100_H_PTD , Water_Produce_Both= "Produce")
df_prod_High["50%"] = get_dec_rate(df= Scen_50_H_PTD , Water_Produce_Both= "Produce")
df_prod_High["25%"] = get_dec_rate(df= Scen_25_H_PTD , Water_Produce_Both= "Produce")
df_prod_High["12.5%"] = get_dec_rate(df= Scen_12_H_PTD , Water_Produce_Both= "Produce")
df_prod_High["6.2%"] = get_dec_rate(df= Scen_6_H_PTD , Water_Produce_Both= "Produce")
df_prod_High["3.1%"] = get_dec_rate(df= Scen_3_H_PTD , Water_Produce_Both= "Produce")
df_prod_High["1.5%"] = get_dec_rate(df= Scen_1_H_PTD , Water_Produce_Both= "Produce")
df_prod_High["0.75%"] = get_dec_rate(df= Scen_075_H_PTD , Water_Produce_Both= "Produce")
df_prod_High["0.4%"] = get_dec_rate(df= Scen_04_H_PTD , Water_Produce_Both= "Produce")
df_prod_High["0.2%"] = get_dec_rate(df= Scen_02_H_PTD , Water_Produce_Both= "Produce")
df_prod_High["0.1%"] = get_dec_rate(df= Scen_01_H_PTD , Water_Produce_Both= "Produce")


df_prod_High.to_csv("C://Users/gareyes3/Documents/GitHub/CPS-Farm-To-Facility-Cilantro/ProductTestingClustersHigh.csv")

#%%

det_rate = []
for i in list(range(1,101,3)):
    x = Process_Model(Days_per_season = 45,
                      Niterations= 1000,
                      Cont_Scenario = 2,#Random Cont Event
                      Testing_Scenario=1,#Sampling at given day
                      #Contamination Information
                      OO_per_L =20,
                      #Water Testing Options
                      Sampling_every_Days_Water = 1, #1 for sampling every day
                      Sampling_every_Days_Product = 1, #as fault 
                      #Testing Options
                      Testing_Day_Water = [0], 
                      Testing_Day_Product = [0],#testing water on day 1
                      Per_Cont_Field = i,
                      Water_Sampling = 0,
                      Product_Sampling_PH = 1,
                      Product_Testing_H = 0
                      )
    det_rate.append(get_dec_rate(df= x , Water_Produce_Both= "Produce"))   

Detection_Rates_High_DPT_List = [item for items in det_rate for item in items]
Detection_Rates_High_DPT = pd.DataFrame({"Drates": Detection_Rates_High_DPT_List,
                                          "Cluster": list(range(1,101,3))})

Detection_Rates_High_DPT.to_csv("C://Users/gareyes3/Documents/GitHub/CPS-Farm-To-Facility-Cilantro/Detection_Rates_High_DPT.csv")


#Propduct end of day
det_rate_FPT = []
for i in list(range(1,101,1)):
    x = Process_Model(Days_per_season = 45,
                      Niterations= 1000,
                      Cont_Scenario = 2,#Random Cont Event
                      Testing_Scenario=2,#Sampling at given day
                      #Contamination Information
                      OO_per_L =20,
                      #Water Testing Options
                      Sampling_every_Days_Water = 1, #1 for sampling every day
                      Sampling_every_Days_Product = 1, #as fault 
                      #Testing Options
                      Testing_Day_Water = [0], 
                      Testing_Day_Product = [45],#testing water on day 1
                      Per_Cont_Field = i,
                      Water_Sampling = 0,
                      Product_Sampling_PH = 1,
                      Product_Testing_H = 0
                      )
    det_rate.append(get_dec_rate(df= x , Water_Produce_Both= "Produce"))   

Detection_Rates_High_FPT_List = [item for items in det_rate_FPT for item in items]
Detection_Rates_High_FPT = pd.DataFrame({"Drates": Detection_Rates_High_FPT_List,
                                          "Cluster": list(range(1,101,1))})

Detection_Rates_High_FPT.to_csv("C://Users/gareyes3/Documents/GitHub/CPS-Farm-To-Facility-Cilantro/Detection_Rates_High_FPT.csv")







#Product Testing end of season
Scen_100_L_PT1 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=2,#Sampling at given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [45],#testing water on day 1
                  Per_Cont_Field = 100,
                  Water_Sampling = 1,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

np.mean(Scen_100_L_PT1[2].sum(axis = 1)>0)

 ##High
#Product Testing every day 
Scen_100_H_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 100,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

np.mean(Scen_100_H_PTD[2].sum(axis = 1)>0)

#Product Testing end of season
Scen_100_H_PT1 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=2,#Sampling at given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [45],#testing water on day 1
                  Per_Cont_Field = 100,
                  Water_Sampling = 1,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

np.mean(Scen_100_H_PT1[2].sum(axis = 1)>0)


#10% Cluster
 ##LOW 
#Product Testing every day 
Scen_10_L_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 10,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

np.mean(Scen_10_L_PTD[2].sum(axis = 1)>0)

#Product Testing end of season
Scen_10_L_PT1 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=2,#Sampling at given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [45],#testing water on day 1
                  Per_Cont_Field = 10,
                  Water_Sampling = 1,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

np.mean(Scen_10_L_PT1[2].sum(axis = 1)>0)


 ##High
#Product Testing every day 
Scen_10_H_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 10,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

np.mean(Scen_10_H_PTD[2].sum(axis = 1)>0)

#Product Testing end of season
Scen_10_H_PT1 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=2,#Sampling at given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [45],#testing water on day 45
                  Per_Cont_Field = 10,
                  Water_Sampling = 1,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )


#1% Cluster
 ##LOW 
#Product Testing every day 
Scen_1_L_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 1,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

#Product Testing end of season
Scen_1_L_PT1 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=2,#Sampling at given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [45],#testing water on day 1
                  Per_Cont_Field = 1,
                  Water_Sampling = 1,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

 ##High
#Product Testing every day 
Scen_1_H_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 1,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

#Product Testing end of season
Scen_1_H_PT1 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=2,#Sampling at given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [45],#testing water on day 45
                  Per_Cont_Field = 1,
                  Water_Sampling = 1,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )


#0.1% Cluster
 ##LOW 
#Product Testing every day 
Scen_01_L_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 0.1,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

#Product Testing end of season
Scen_01_L_PT1 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=2,#Sampling at given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [45],#testing water on day 1
                  Per_Cont_Field = 0.1,
                  Water_Sampling = 1,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

 ##High
#Product Testing every day 
Scen_01_H_PTD = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=1,#Sampling at given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [0],#testing water on day 1
                  Per_Cont_Field = 0.1,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )

#Product Testing end of season
Scen_01_H_PT1 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 1000,
                  Cont_Scenario = 2,#Random Cont Event
                  Testing_Scenario=2,#Sampling at given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [45],#testing water on day 45
                  Per_Cont_Field = 0.1,
                  Water_Sampling = 1,
                  Product_Sampling_PH = 1,
                  Product_Testing_H = 0
                  )



#Summary DF:
col_name_prod=["Daily Produce Testing 100%",
               "End of season produce testing 100%",
               "Daily Produce Testing 10%",
               "End of season produce testing 10%",
               "Daily Produce Testing 1%",
               "End of season produce testing 1%",
               "Daily Produce Testing 0.1%",
               "End of season produce testing 0.1%"
               ]

#LOW:
df_prod_Low = pd.DataFrame(columns=[col_name_prod])

df_prod_Low["Daily Produce Testing 100%"] = get_pdetect(df= Scen_100_L_PTD , Water_Produce_Both= "Produce")
df_prod_Low["End of season produce testing 100%"] = get_pdetect(df= Scen_100_L_PT1 , Water_Produce_Both= "Produce")
df_prod_Low["Daily Produce Testing 10%"] = get_pdetect(df= Scen_10_L_PTD , Water_Produce_Both= "Produce")
df_prod_Low["End of season produce testing 10%"] = get_pdetect(df= Scen_10_L_PT1 , Water_Produce_Both= "Produce")
df_prod_Low["Daily Produce Testing 1%"] = get_pdetect(df= Scen_1_L_PTD , Water_Produce_Both= "Produce")
df_prod_Low["End of season produce testing 1%"] = get_pdetect(df= Scen_1_L_PT1 , Water_Produce_Both= "Produce")
df_prod_Low["Daily Produce Testing 0.1%"] = get_pdetect(df= Scen_01_L_PTD , Water_Produce_Both= "Produce")
df_prod_Low["End of season produce testing 0.1%"] = get_pdetect(df= Scen_01_L_PT1 , Water_Produce_Both= "Produce")

#np.mean(get_pdetect(df= Scen_F1_L_PTD , Water_Produce_Both= "Produce"))


#HIGH
df_prod_High = pd.DataFrame(columns=[col_name_prod])
df_prod_High["Daily Produce Testing 100%"] = get_pdetect(df= Scen_100_H_PTD , Water_Produce_Both= "Produce")
df_prod_High["End of season produce testing 100%"] = get_pdetect(df= Scen_100_H_PT1 , Water_Produce_Both= "Produce")
df_prod_High["Daily Produce Testing 10%"] = get_pdetect(df= Scen_10_H_PTD , Water_Produce_Both= "Produce")
df_prod_High["End of season produce testing 10%"] = get_pdetect(df= Scen_10_H_PT1 , Water_Produce_Both= "Produce")
df_prod_High["Daily Produce Testing 1%"] = get_pdetect(df= Scen_1_H_PTD , Water_Produce_Both= "Produce")
df_prod_High["End of season produce testing 1%"] = get_pdetect(df= Scen_1_H_PT1 , Water_Produce_Both= "Produce")
df_prod_High["Daily Produce Testing 0.1%"] = get_pdetect(df= Scen_01_H_PTD , Water_Produce_Both= "Produce")
df_prod_High["End of season produce testing 0.1%"] = get_pdetect(df= Scen_01_H_PT1 , Water_Produce_Both= "Produce")




df_prod_Low.to_csv("C://Users/gareyes3/Documents/GitHub/CPS-Farm-To-Facility-Cilantro/ProductTestingClusterslow.csv")
df_prod_High.to_csv("C://Users/gareyes3/Documents/GitHub/CPS-Farm-To-Facility-Cilantro/ProductTestingClustersHigh.csv")



get_dec_rate

df_prod_Low = pd.DataFrame(columns=[col_name_prod])

df_prod_Low["Daily Produce Testing 100%"] = get_dec_rate(df= Scen_100_L_PTD , Water_Produce_Both= "Produce")
df_prod_Low["End of season produce testing 100%"] = get_dec_rate(df= Scen_100_L_PT1 , Water_Produce_Both= "Produce")
df_prod_Low["Daily Produce Testing 10%"] = get_dec_rate(df= Scen_10_L_PTD , Water_Produce_Both= "Produce")
df_prod_Low["End of season produce testing 10%"] = get_dec_rate(df= Scen_10_L_PT1 , Water_Produce_Both= "Produce")
df_prod_Low["Daily Produce Testing 1%"] = get_dec_rate(df= Scen_1_L_PTD , Water_Produce_Both= "Produce")
df_prod_Low["End of season produce testing 1%"] = get_dec_rate(df= Scen_1_L_PT1 , Water_Produce_Both= "Produce")
df_prod_Low["Daily Produce Testing 0.1%"] = get_dec_rate(df= Scen_01_L_PTD , Water_Produce_Both= "Produce")
df_prod_Low["End of season produce testing 0.1%"] = get_dec_rate(df= Scen_01_L_PT1 , Water_Produce_Both= "Produce")

df_prod_Low.to_csv("C://Users/gareyes3/Documents/GitHub/CPS-Farm-To-Facility-Cilantro/ProductTestingClusterslow.csv")


df_prod_Low = pd.DataFrame(columns=[col_name_prod])

df_prod_Low["Daily Produce Testing 100%"] =get_dec_rate(df= Scen_100_L_PTD , Water_Produce_Both= "Produce")
df_prod_Low["End of season produce testing 100%"] = get_dec_rate(df= Scen_100_L_PT1 , Water_Produce_Both= "Produce")
df_prod_Low["Daily Produce Testing 10%"] = get_dec_rate(df= Scen_10_L_PTD , Water_Produce_Both= "Produce")
df_prod_Low["End of season produce testing 10%"] = get_dec_rate(df= Scen_10_L_PT1 , Water_Produce_Both= "Produce")
df_prod_Low["Daily Produce Testing 1%"] = get_dec_rate(df= Scen_1_L_PTD , Water_Produce_Both= "Produce")
df_prod_Low["End of season produce testing 1%"] = get_dec_rate(df= Scen_1_L_PT1 , Water_Produce_Both= "Produce")
df_prod_Low["Daily Produce Testing 0.1%"] = get_dec_rate(df= Scen_01_L_PTD , Water_Produce_Both= "Produce")
df_prod_Low["End of season produce testing 0.1%"] = get_dec_rate(df= Scen_01_L_PT1 , Water_Produce_Both= "Produce")


df_prod_High.to_csv("C://Users/gareyes3/Documents/GitHub/CPS-Farm-To-Facility-Cilantro/ProductTestingClustersHigh.csv")





import seaborn as sns

Baseline_1[1].melt()   
        
        
sns.lineplot(data= Scen_B1_H_DTW[1].melt() , x = "variable", y = "value")    
        
    
#Best Day product testing
Days_Positive = Baseline_1[2].melt()
Days_Positive = Days_Positive[Days_Positive["value"] == 1]
sns.histplot(data = Days_Positive, x = "variable",bins = 45)
    
    
#Best Day water testing
Days_Positive = Baseline_1[0].melt()
Days_Positive = Days_Positive[Days_Positive["value"] == 1]
sns.histplot(data = Days_Positive, x = "variable", bins = 45) 
    
sum(Scen_B2_L_WT1[0][1]>0)

66520000.0-((122/10_000)*66520000.0)

