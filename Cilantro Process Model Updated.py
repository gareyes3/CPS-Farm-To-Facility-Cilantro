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
    sampled_oo_l = []
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
            Sampled_OO = 1
        else:
            Pr_Detect = 0  
            Sampled_OO = 0
        probs_detection.append(Pr_Detect)
        if np.random.uniform(0,1) < Pr_Detect:
            sample_resuls.append(1)
            df2.loc[Sample_Indeces, 'PositiveSamples'] = df2.loc[Sample_Indeces, 'PositiveSamples'] + 1
        else: 
            sample_resuls.append(0)
        sampled_oo_l.append(Sampled_OO)
        samp_more_1_o = sum(sampled_oo_l)/N_25g_Samples
    if sum(sample_resuls)> 0:
        reject_YN = 1
    else:
        reject_YN = 0
        
    if len(probs_detection)>1:
        pdetect = 1-np.prod([1-i for i in probs_detection])
    else: 
        pdetect = probs_detection[0]
    return [df2, reject_YN, pdetect, samp_more_1_o]

'''
#Creating the Field
Field_Yield = 22_000 #lb
Plant_Weight = 1 #lb
Total_Plants = int(Field_Yield/Plant_Weight)
Total_Plants_List = range(1,Total_Plants+1)

Cilantro_df=pd.DataFrame({"Plant_ID": Total_Plants_List,
                       "Weight": Plant_Weight,
                       "Case_PH": 0,
                       "Oo": 0,
                       "Oo_BRej":"",
                       "Location": 1,
                       'PositiveSamples':0,
                       "Rej_Acc" :"Acc"
                  })

Cilantro_df = field_cont_percetage2(df = Cilantro_df, 
                                    percent_cont = 10, 
                                    Hazard_lvl =200000,
                                    No_Cont_Clusters = 1)


Cilantro_Sampling_25g(df = Cilantro_df,
                      Sample_Weight = 25,
                      N_25g_Samples = 100,
                      N_Grabs_Sample =10,
                      Plant_Weight = 1, 
                      loaded_model = qPCR_Model )
'''

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
#filename_qPCR = 'C://Users/Gustavo Reyes/Documents/GitHubFiles/CPS-Farm-To-Facility-Cilantro/logistic_AW_Testing_qPCR.sav'

qPCR_Model_AW = pickle.load(open(filename_qPCR, 'rb'))
qPCR_Model_AW.predict_proba(np.array([20]).reshape(-1,1))[0][1] #from logistic

filename_qPCR = 'C://Users/gareyes3/Documents/GitHub/CPS-Farm-To-Facility-Cilantro/logistic_Prod_Test_qPCR_FDA.sav'
#filename_qPCR = 'C://Users/Gustavo Reyes/Documents/GitHubFiles/CPS-Farm-To-Facility-Cilantro/logistic_Prod_Test_qPCR_FDA.sav'

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
Total_L_Season = 63.6*63.6*(0.0254*Water_Irrigation_In)*1000 # field yield 40.46m2 * 0.348 m of water * 10000 to convert tot m3
Days_per_season = 45 #days
Irrigation_Days_per_season = 45
L_water_day = Total_L_Season/Irrigation_Days_per_season


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
                  Product_Testing_H = 0, ##1 is on 0 is off
                  #sampling
                  N_Samples_Prod = 1,
                  N_Grabs_Prod = 1
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
    #tells if anything was sampled
    Produce_sampledYN_DF = Output_DF_Creation(Column_Names =np.arange(1,Days_per_season+1), Niterations= Niterations)

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
            Random_Irr_Day_Scen2 = random.randint(1,Days_per_season) #here
            
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
                Irrigation_Levels_Days = Initial_Levels_Bulk/Days_per_season #here
            elif Cont_Scenario ==1:
                Initial_Levels_Bulk = int(Total_L_Season*OO_per_L)
                Irrigation_Levels_Days = Initial_Levels_Bulk/Days_per_season #here
        
        
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
                                          N_25g_Samples = N_Samples_Prod,
                                          N_Grabs_Sample = N_Grabs_Prod ,
                                          Plant_Weight = Plant_Weight, 
                                          loaded_model =  qPCR_Model)
              
                  Produce_Outcome_DF = Output_Collection_any_output(outputDF = Produce_Outcome_DF, Step_Column = i,iteration=k, outcome = Produce_test_results[1])
                  Produce_PrRej_DF = Output_Collection_any_output(outputDF = Produce_PrRej_DF, Step_Column = i,iteration=k, outcome = Produce_test_results[2])
                  Produce_sampledYN_DF = Output_Collection_any_output(outputDF = Produce_sampledYN_DF, Step_Column = i,iteration=k, outcome = Produce_test_results[3])
                  Contam_Produce_DF=Output_Collection_ProduceCont(df=Cilantro_df, outputDF=Contam_Produce_DF, Step_Column = i,iteration = k)
        
        
            #Harvest Sampling
            if i == Days_per_season:
                #Harvest Testing
                if Product_Testing_H  == 1:
                    Harvest_Test_Results =Cilantro_Sampling_25g(df=Cilantro_df,
                                          Sample_Weight = Sample_Weight,
                                          N_25g_Samples =N_Samples_Prod,
                                          N_Grabs_Sample = N_Grabs_Prod ,
                                          Plant_Weight = Plant_Weight, 
                                          loaded_model =  qPCR_Model)
                    
                    #Cilantro_df =F_Rejection_Rule_C (df= Cilantro_df )
                
                    Harvest_Sampling_Outcome_DF = Output_Collection_any_output(outputDF = Harvest_Sampling_Outcome_DF, Step_Column = i,iteration=k, outcome =  Harvest_Test_Results[1])
                    Harvest_Sampling_PrRej_DF = Output_Collection_any_output(outputDF = Harvest_Sampling_PrRej_DF, Step_Column = i,iteration=k, outcome =  Harvest_Test_Results[2])
                    Contam_HS_DF=Output_Collection_ProduceCont(df=Cilantro_df, outputDF=Contam_HS_DF, Step_Column = i,iteration = k)
            
            Final_CFUS_DF=Output_Collection_ProduceCont(df=Cilantro_df, outputDF=Final_CFUS_DF, Step_Column = i,iteration = k)

    return [Water_Outcome_DF,Water_PrRej_DF,Produce_Outcome_DF,Produce_PrRej_DF, Contam_Produce_DF,Harvest_Sampling_Outcome_DF,Harvest_Sampling_PrRej_DF,Contam_HS_DF,Final_CFUS_DF, Produce_sampledYN_DF]

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
                  Niterations= 10000,
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

#B1_Water testing 1 time per season at the end of the season - Low
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
                  Testing_Day_Water = [45], #testing water on day 1
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 0, 
                  Product_Testing_H = 0
                  )

#B1_Water testing 1 time per season at the end of the season - High
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
                  Testing_Day_Water = [45], #testing water on day 1
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 0, 
                  Product_Testing_H = 0
                  )

#B1_Water testing 1 time per season at the end of the season - Low
Scen_B1_L_WT4 = Process_Model(
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

#B1_Water testing 1 time per season at the end of the season - High
Scen_B1_H_WT4 = Process_Model(
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

#B1_Water testing 2 times per season (end and mid) - Low
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
#B1_Water testing 2 times per season (end and mid) - High
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

#B1_Water testing 3 times per season (end,mid,start) - Low
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

#B1_Water testing 3 times per season (end,mid,start) - High
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


#B1_product testing 1 time per season at the end of the season - Low
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
                  Testing_Day_Product = [45],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

#B1_product testing 1 time per season at the end of the season - High
Scen_B1_H_PT1 = Process_Model(
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
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [45],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

#B1_product testing 1 time per season at the end of the season - Low
Scen_B1_L_PT4 = Process_Model(
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

#B1_product testing 1 time per season at the end of the season - High
Scen_B1_H_PT4 = Process_Model(
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
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [1],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

#B1_product testing 2 times per season end and mid season - Low
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
#B1_product testing 2 times per season end and mid season - High
Scen_B1_H_PT2 = Process_Model(
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
                  Niterations= 10000,
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

#B2_Water testing 1 time per season at the end of the season - Low
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
                  Testing_Day_Water = [45], #testing water on day 1
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
                  Testing_Day_Water = [45], #testing water on day 1
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 0, 
                  Product_Testing_H = 0
                  )

#B2_Water testing 1 time per season at the end of the season - Low
Scen_B2_L_WT4 = Process_Model(
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
Scen_B2_H_WT4 = Process_Model(
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
                  Testing_Day_Water = [22,45], #testing water on day 1
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
                  Testing_Day_Water = [22,45], #testing water on day 1
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
                  Testing_Day_Product = [45],
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
                  Testing_Day_Product = [45],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

#B2_product testing 1 time per season at the start of the season - Low
Scen_B2_L_PT4 = Process_Model(
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
Scen_B2_H_PT4 = Process_Model(
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


#B1_Water testing 1 time per season at the start of the season - High
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
                  Testing_Day_Water = [45],
                  Testing_Day_Product = [45],
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
                  Testing_Day_Water = [45], #testing water on day 1
                  Testing_Day_Product = [45],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )


Scen_B1_H_WPT4 = Process_Model(
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
                  Testing_Day_Product = [1],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

Scen_B1_L_WPT4 = Process_Model(
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
#----
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
                  Testing_Day_Water = [45],
                  Testing_Day_Product = [45],
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
                  Testing_Day_Water = [45], #testing water on day 1
                  Testing_Day_Product = [45],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )



Scen_B2_H_WPT4 = Process_Model(
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
Scen_B2_L_WPT4 = Process_Model(
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
        return [np.mean(data.sum(axis = 1)>0)]
    
    elif Water_Produce_Both == "Produce":
        data = df[2]
        return [np.mean(data.sum(axis = 1)>0)]
    elif Water_Produce_Both == "Both":
        data_W = df[0]
        data_P = df[2]
        x_1 = data_W.sum(axis = 1)>0
        z_2 = data_P.sum(axis = 1)>0
        resuls_app = []
        for i in range(0,len(x_1)):
            if (int(x_1[i]) + int( z_2[i])>0):
                resuls_app.append(1)
            else:
               resuls_app.append(0) 
        resuls_app
        return(np.mean(resuls_app))
    
        

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



#Getting detection rates:

    #Baseline
get_dec_rate(df  = Out_B1_L, Water_Produce_Both = "Both")
get_dec_rate(df  = Out_B1_H, Water_Produce_Both = "Both")
get_dec_rate(df  = Out_B2_L, Water_Produce_Both = "Both")
get_dec_rate(df  = Out_B2_H, Water_Produce_Both = "Both")

    #Baseline 1 - High
get_dec_rate(df  = Scen_B1_H_DTP, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B1_H_DTW, Water_Produce_Both = "Water")

get_dec_rate(df  = Scen_B1_H_WT1, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B1_H_WT2, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B1_H_WT3, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B1_H_WT4, Water_Produce_Both = "Water")

get_dec_rate(df  = Scen_B1_H_PT1, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B1_H_PT2, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B1_H_PT3, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B1_H_PT4, Water_Produce_Both = "Produce")


get_dec_rate(df  = Scen_B1_H_DWPT, Water_Produce_Both = "Both")
get_dec_rate(df  = Scen_B1_H_WPT1, Water_Produce_Both = "Both")
get_dec_rate(df  = Scen_B1_H_WPT2, Water_Produce_Both = "Both") 
get_dec_rate(df  = Scen_B1_H_WPT3, Water_Produce_Both = "Both") 
get_dec_rate(df  = Scen_B1_H_WPT4, Water_Produce_Both = "Both")

    #Baseline 1 - Low
get_dec_rate(df  = Scen_B1_L_DTP, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B1_L_DTW, Water_Produce_Both = "Water")

get_dec_rate(df  = Scen_B1_L_WT1, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B1_L_WT2, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B1_L_WT3, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B1_L_WT4, Water_Produce_Both = "Water")

get_dec_rate(df  = Scen_B1_L_PT1, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B1_L_PT2, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B1_L_PT3, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B1_L_PT4, Water_Produce_Both = "Produce")


get_dec_rate(df  = Scen_B1_L_DWPT, Water_Produce_Both = "Both")
get_dec_rate(df  = Scen_B1_L_WPT1, Water_Produce_Both = "Both")
get_dec_rate(df  = Scen_B1_L_WPT2, Water_Produce_Both = "Both") 
get_dec_rate(df  = Scen_B1_L_WPT3, Water_Produce_Both = "Both") 
get_dec_rate(df  = Scen_B1_L_WPT4, Water_Produce_Both = "Both")



    #Baseline 2 - High
get_dec_rate(df  = Scen_B2_H_DTP, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B2_H_DTW, Water_Produce_Both = "Water")

get_dec_rate(df  = Scen_B2_H_WT1, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B2_H_WT2, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B2_H_WT3, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B2_H_WT4, Water_Produce_Both = "Water")

get_dec_rate(df  = Scen_B2_H_PT1, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B2_H_PT2, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B2_H_PT3, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B2_H_PT4, Water_Produce_Both = "Produce")


get_dec_rate(df  = Scen_B2_H_DWPT, Water_Produce_Both = "Both")
get_dec_rate(df  = Scen_B2_H_WPT1, Water_Produce_Both = "Both") 
get_dec_rate(df  = Scen_B2_H_WPT2, Water_Produce_Both = "Both") 
get_dec_rate(df  = Scen_B2_H_WPT3, Water_Produce_Both = "Both") 
get_dec_rate(df  = Scen_B2_H_WPT4, Water_Produce_Both = "Both") 

    #Baseline 2 - Low
get_dec_rate(df  = Scen_B2_L_DTP, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B2_L_DTW, Water_Produce_Both = "Water")

get_dec_rate(df  = Scen_B2_L_WT1, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B2_L_WT2, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B2_L_WT3, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B2_L_WT4, Water_Produce_Both = "Water")

get_dec_rate(df  = Scen_B2_L_PT1, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B2_L_PT2, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B2_L_PT3, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B2_L_PT4, Water_Produce_Both = "Produce")


get_dec_rate(df  = Scen_B2_L_DWPT, Water_Produce_Both = "Both") #
get_dec_rate(df  = Scen_B2_L_WPT1, Water_Produce_Both = "Both") #
get_dec_rate(df  = Scen_B2_L_WPT2, Water_Produce_Both = "Both") 
get_dec_rate(df  = Scen_B2_L_WPT3, Water_Produce_Both = "Both") 
get_dec_rate(df  = Scen_B2_L_WPT4, Water_Produce_Both = "Both")#
    



#%%

def Extracting_outputs_sample_prob(df):
    probability_of_sample = np.mean(df[9].sum(axis = 1)>0)
    df_narrow= df[3][df[9].sum(axis = 1)>0]
    
    Pdetects = []
    for i in list(df_narrow.index):
        Probs = df_narrow.loc[[i]].values.tolist()[0]
        Probs_g_zero = [i for i in Probs if i >0]
        if len(Probs_g_zero)>0:
            Probs_g_zero_rem = [1-i for i in Probs_g_zero]
            Prod_Probs_g_zero_rem=np.prod(Probs_g_zero_rem)
            Pdetect = 1-Prod_Probs_g_zero_rem
        else:
            Pdetect = 0 
        Pdetects.append(Pdetect)
    P_detect_pres = np.mean(Pdetects)
    return [probability_of_sample,P_detect_pres]

#### Scenarios

list_of_clusters = []
for i in list(range(11)):
    list_of_clusters.append(100/2**i)

det_rate = []
samp_rate = []
assay_rate = []
for i in list_of_clusters:
    x = Process_Model(Days_per_season = 45,
                      Niterations= 10000,
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
    samp_rate.append(Extracting_outputs_sample_prob(x)[0])
    assay_rate.append(Extracting_outputs_sample_prob(x)[1])

Detection_Rates_High_DPT_List = [item for items in det_rate for item in items]
Detection_Rates_High_DPT = pd.DataFrame({"Drates": Detection_Rates_High_DPT_List,
                                          "Cluster": list_of_clusters,
                                          "samp_rate":samp_rate,
                                          "assay_rate":assay_rate})

Detection_Rates_High_DPT.to_csv("C://Users/Gustavo Reyes/Documents/GitHubFiles/CPS-Farm-To-Facility-Cilantro/Detection_Rates_High_DPT_LOC-r2.csv")

##############################################################################

#Propduct end of day
det_rate_FPT = []
samp_rate_FTP = []
assay_rate_FTP = []
for i in list_of_clusters :
    x = Process_Model(Days_per_season = 45,
                      Niterations= 10000,
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
    det_rate_FPT.append(get_dec_rate(df= x , Water_Produce_Both= "Produce")) 
    samp_rate_FTP.append(Extracting_outputs_sample_prob(x)[0])
    assay_rate_FTP.append(Extracting_outputs_sample_prob(x)[1])

Detection_Rates_High_FPT_List = [item for items in det_rate_FPT for item in items]
Detection_Rates_High_FPT = pd.DataFrame({"Drates": Detection_Rates_High_FPT_List,
                                          "Cluster": list_of_clusters,
                                          "samp_rate":samp_rate_FTP,
                                          "assay_rate":assay_rate_FTP})

Detection_Rates_High_FPT.to_csv("C://Users/Gustavo Reyes/Documents/GitHubFiles/CPS-Farm-To-Facility-Cilantro/Detection_Rates_High_FPT_LOC-r2.csv")



#Propduct end of day
det_rate_DPT_L = []
samp_rate_DPT_L =[]
assay_rate_DPT_L = []
for i in list_of_clusters :
    x = Process_Model(Days_per_season = 45,
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
                      Per_Cont_Field = i,
                      Water_Sampling = 0,
                      Product_Sampling_PH = 1,
                      Product_Testing_H = 0
                      )
    det_rate_DPT_L.append(get_dec_rate(df= x , Water_Produce_Both= "Produce"))  
    samp_rate_DPT_L.append(Extracting_outputs_sample_prob(x)[0])
    assay_rate_DPT_L.append(Extracting_outputs_sample_prob(x)[1])

Detection_Rates_Low_DPT = [item for items in det_rate_DPT_L for item in items]
Detection_Rates_Low_DPT = pd.DataFrame({"Drates": Detection_Rates_Low_DPT,
                                          "Cluster": list_of_clusters,
                                          "samp_rate":samp_rate_DPT_L,
                                          "assay_rate":assay_rate_DPT_L})


Detection_Rates_Low_DPT.to_csv("C://Users/Gustavo Reyes/Documents/GitHubFiles/CPS-Farm-To-Facility-Cilantro/Detection_Rates_Low_DPT_LOC-r2.csv")




#Propduct end of day
det_rate_FPT_L = []
samp_rate_FPT_L =[]
assay_rate_FPT_L = []
for i in list_of_clusters :
    x = Process_Model(Days_per_season = 45,
                      Niterations= 10000,
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
                      Per_Cont_Field = i,
                      Water_Sampling = 0,
                      Product_Sampling_PH = 1,
                      Product_Testing_H = 0
                      )
    det_rate_FPT_L.append(get_dec_rate(df= x , Water_Produce_Both= "Produce"))  
    samp_rate_FPT_L.append(Extracting_outputs_sample_prob(x)[0])
    assay_rate_FPT_L.append(Extracting_outputs_sample_prob(x)[1])

Detection_Rates_Low_FPT = [item for items in det_rate_FPT_L for item in items]
Detection_Rates_Low_FPT = pd.DataFrame({"Drates": Detection_Rates_Low_FPT,
                                          "Cluster": list_of_clusters,
                                          "samp_rate":samp_rate_FPT_L,
                                          "assay_rate":assay_rate_FPT_L})


Detection_Rates_Low_FPT.to_csv("C://Users/Gustavo Reyes/Documents/GitHubFiles/CPS-Farm-To-Facility-Cilantro/Detection_Rates_Low_FPT_LOC-r2.csv")



#Now Doing 10 samples of 25g

#Propduct end of day
det_rate_FPT_10s_L = []
samp_rate_FPT_10s_L =[]
assay_rate_FPT_10s_L = []
for i in list_of_clusters :
    x = Process_Model(Days_per_season = 45,
                      Niterations= 10000,
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
                      Per_Cont_Field = i,
                      Water_Sampling = 0,
                      Product_Sampling_PH = 1,
                      Product_Testing_H = 0,
                      #sampling
                      N_Samples_Prod = 10,
                      N_Grabs_Prod = 1,
                      )
    det_rate_FPT_10s_L.append(get_dec_rate(df= x , Water_Produce_Both= "Produce"))  
    samp_rate_FPT_10s_L.append(Extracting_outputs_sample_prob(x)[0])
    assay_rate_FPT_10s_L.append(Extracting_outputs_sample_prob(x)[1])

Detection_Rates_Low_10s_FPT = [item for items in det_rate_FPT_10s_L for item in items]
Detection_Rates_Low_10s_FPT = pd.DataFrame({"Drates": Detection_Rates_Low_10s_FPT,
                                          "Cluster": list_of_clusters,
                                          "samp_rate":samp_rate_FPT_10s_L,
                                          "assay_rate":assay_rate_FPT_10s_L})


Detection_Rates_Low_10s_FPT.to_csv("C://Users/Gustavo Reyes/Documents/GitHubFiles/CPS-Farm-To-Facility-Cilantro/Detection_Rates_Low_10s_FPT_LOC-r2.csv")


#Now Doing 10 samples of 25g

#Propduct end of day
det_rate_FPT_10s_H = []
samp_rate_FPT_10s_H =[]
assay_rate_FPT_10s_H = []
for i in list_of_clusters :
    x = Process_Model(Days_per_season = 45,
                      Niterations= 10000,
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
                      Product_Testing_H = 0,
                      #sampling
                      N_Samples_Prod = 10,
                      N_Grabs_Prod = 1
                      )
    det_rate_FPT_10s_H.append(get_dec_rate(df= x , Water_Produce_Both= "Produce"))  
    samp_rate_FPT_10s_H.append(Extracting_outputs_sample_prob(x)[0])
    assay_rate_FPT_10s_H.append(Extracting_outputs_sample_prob(x)[1])

Detection_Rates_High_10s_FPT = [item for items in det_rate_FPT_10s_H for item in items]
Detection_Rates_High_10s_FPT = pd.DataFrame({"Drates": Detection_Rates_High_10s_FPT,
                                          "Cluster": list_of_clusters,
                                          "samp_rate":samp_rate_FPT_10s_H,
                                          "assay_rate":assay_rate_FPT_10s_H})


Detection_Rates_High_10s_FPT.to_csv("C://users/Gustavo Reyes/Documents/GitHubFiles/CPS-Farm-To-Facility-Cilantro/Detection_Rates_High_10s_FPT_LOC-r2.csv")



#Now Doing 10 samples of 25g

#Propduct end of day
det_rate_FPT_45s_H = []
samp_rate_FPT_45s_H =[]
assay_rate_FPT_45s_H = []
for i in list_of_clusters :
    x = Process_Model(Days_per_season = 45,
                      Niterations= 10000,
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
                      Product_Testing_H = 0,
                      #sampling
                      N_Samples_Prod = 45,
                      N_Grabs_Prod = 1
                      )
    det_rate_FPT_45s_H.append(get_dec_rate(df= x , Water_Produce_Both= "Produce"))  
    samp_rate_FPT_45s_H.append(Extracting_outputs_sample_prob(x)[0])
    assay_rate_FPT_45s_H.append(Extracting_outputs_sample_prob(x)[1])

Detection_Rates_High_45s_FPT = [item for items in det_rate_FPT_45s_H for item in items]
Detection_Rates_High_45s_FPT = pd.DataFrame({"Drates": Detection_Rates_High_45s_FPT,
                                          "Cluster": list_of_clusters,
                                          "samp_rate":samp_rate_FPT_45s_H,
                                          "assay_rate":assay_rate_FPT_45s_H})


Detection_Rates_High_45s_FPT.to_csv("C://Users/Gustavo Reyes/Documents/GitHubFiles/CPS-Farm-To-Facility-Cilantro/Detection_Rates_High_45s_FPT_LOC-r2.csv")



#Now Doing 10 samples of 25g

#Propduct end of day
det_rate_FPT_45s_L = []
samp_rate_FPT_45s_L =[]
assay_rate_FPT_45s_L = []
for i in list_of_clusters :
    x = Process_Model(Days_per_season = 45,
                      Niterations= 10000,
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
                      Per_Cont_Field = i,
                      Water_Sampling = 0,
                      Product_Sampling_PH = 1,
                      Product_Testing_H = 0,
                      #sampling
                      N_Samples_Prod = 45,
                      N_Grabs_Prod = 1
                      )
    det_rate_FPT_45s_L.append(get_dec_rate(df= x , Water_Produce_Both= "Produce"))  
    samp_rate_FPT_45s_L.append(Extracting_outputs_sample_prob(x)[0])
    assay_rate_FPT_45s_L.append(Extracting_outputs_sample_prob(x)[1])

Detection_Rates_Low_45s_FPT = [item for items in det_rate_FPT_45s_L for item in items]
Detection_Rates_Low_45s_FPT = pd.DataFrame({"Drates": Detection_Rates_Low_45s_FPT,
                                          "Cluster": list_of_clusters,
                                          "samp_rate":samp_rate_FPT_45s_L,
                                          "assay_rate":assay_rate_FPT_45s_L})


Detection_Rates_Low_45s_FPT.to_csv("C://Users//Gustavo Reyes/Documents/GitHubFiles/CPS-Farm-To-Facility-Cilantro/Detection_Rates_Low_45s_FPT_LOC-r2.csv")



Func_Water_Sampling (total_oocyst_bw = 5989600, 
                                bw_volume =249480 , 
                                sample_size_volume =10,
                                total_samples = 1, 
                                loaded_model =qPCR_Model_AW )

Func_Water_Sampling (total_oocyst_bw = 5989600, 
                                bw_volume =249480 , 
                                sample_size_volume =10,
                                total_samples = 1, 
                                loaded_model =qPCR_Model_AW )


#%%

#ADDITIONAL SCENARIOS Irrgation days,yield and water

def Process_Model_Extra(Days_per_season,
                  Niterations,
                  Custom_Irrigation_Days,#only for cot scenario 3
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
                  Product_Testing_H = 0, ##1 is on 0 is off
                  #sampling
                  N_Samples_Prod = 1,
                  N_Grabs_Prod = 1
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
    #tells if anything was sampled
    Produce_sampledYN_DF = Output_DF_Creation(Column_Names =np.arange(1,Days_per_season+1), Niterations= Niterations)

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
            Random_Irr_Day_Scen2 = random.randint(1,Days_per_season) #here
            
        if Cont_Scenario ==32:
            Random_Irr_Day_Scen2 = random.randint(1,Days_per_season) #here
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
            
        
        Days_of_Irrigation = random.sample(range(1,Days_per_season+1), Custom_Irrigation_Days)
        

        
        
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
                Irrigation_Levels_Days = Initial_Levels_Bulk/Days_per_season #here
            if Cont_Scenario ==1:
                Initial_Levels_Bulk = int(Total_L_Season*OO_per_L)
                Irrigation_Levels_Days = Initial_Levels_Bulk/Days_per_season #here
                
            
            if Cont_Scenario ==31 and i in Days_of_Irrigation:
                Initial_Levels_Bulk = int(Total_L_Season*OO_per_L)
                Irrigation_Levels_Days =Irrigation_Levels_Days = Initial_Levels_Bulk/len(Days_of_Irrigation) 
            
            if Cont_Scenario ==31 and i not in Days_of_Irrigation:
                Initial_Levels_Bulk = int(Total_L_Season*OO_per_L)
                Irrigation_Levels_Days = 0
    
            if Cont_Scenario ==32 and i == Random_Irr_Day_Scen2 and i in Days_of_Irrigation:
                Initial_Levels_Bulk = int(Total_L_Season*OO_per_L)
                #print(Initial_Levels_Bulk)
                Irrigation_Levels_Days = Initial_Levels_Bulk/len(Days_of_Irrigation) #Contamination in bulk split into different  levels
                #print(Irrigation_Levels_Days)  
            if Cont_Scenario ==32 and i == Random_Irr_Day_Scen2 and i not in Days_of_Irrigation:
                Initial_Levels_Bulk = int(Total_L_Season*OO_per_L)
                #print(Initial_Levels_Bulk)
                Irrigation_Levels_Days = 0 #Contamination in bulk split into different  levels
                #print(Irrigation_Levels_Days)  
            if Cont_Scenario ==32 and i != Random_Irr_Day_Scen2:
                Initial_Levels_Bulk = 0
                #print(Initial_Levels_Bulk)
                Irrigation_Levels_Days = 0#Contamination in bulk split into different  levels
                #print(Irrigation_Levels_Days)     
            
        
        
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
                                          N_25g_Samples = N_Samples_Prod,
                                          N_Grabs_Sample = N_Grabs_Prod ,
                                          Plant_Weight = Plant_Weight, 
                                          loaded_model =  qPCR_Model)
              
                  Produce_Outcome_DF = Output_Collection_any_output(outputDF = Produce_Outcome_DF, Step_Column = i,iteration=k, outcome = Produce_test_results[1])
                  Produce_PrRej_DF = Output_Collection_any_output(outputDF = Produce_PrRej_DF, Step_Column = i,iteration=k, outcome = Produce_test_results[2])
                  Produce_sampledYN_DF = Output_Collection_any_output(outputDF = Produce_sampledYN_DF, Step_Column = i,iteration=k, outcome = Produce_test_results[3])
                  Contam_Produce_DF=Output_Collection_ProduceCont(df=Cilantro_df, outputDF=Contam_Produce_DF, Step_Column = i,iteration = k)
        
        
            #Harvest Sampling
            if i == Days_per_season:
                #Harvest Testing
                if Product_Testing_H  == 1:
                    Harvest_Test_Results =Cilantro_Sampling_25g(df=Cilantro_df,
                                          Sample_Weight = Sample_Weight,
                                          N_25g_Samples =N_Samples_Prod,
                                          N_Grabs_Sample = N_Grabs_Prod ,
                                          Plant_Weight = Plant_Weight, 
                                          loaded_model =  qPCR_Model)
                    
                    #Cilantro_df =F_Rejection_Rule_C (df= Cilantro_df )
                
                    Harvest_Sampling_Outcome_DF = Output_Collection_any_output(outputDF = Harvest_Sampling_Outcome_DF, Step_Column = i,iteration=k, outcome =  Harvest_Test_Results[1])
                    Harvest_Sampling_PrRej_DF = Output_Collection_any_output(outputDF = Harvest_Sampling_PrRej_DF, Step_Column = i,iteration=k, outcome =  Harvest_Test_Results[2])
                    Contam_HS_DF=Output_Collection_ProduceCont(df=Cilantro_df, outputDF=Contam_HS_DF, Step_Column = i,iteration = k)
            
            Final_CFUS_DF=Output_Collection_ProduceCont(df=Cilantro_df, outputDF=Final_CFUS_DF, Step_Column = i,iteration = k)

    return [Water_Outcome_DF,Water_PrRej_DF,Produce_Outcome_DF,Produce_PrRej_DF, Contam_Produce_DF,Harvest_Sampling_Outcome_DF,Harvest_Sampling_PrRej_DF,Contam_HS_DF,Final_CFUS_DF, Produce_sampledYN_DF]

#%% Snario Stuff

#Cilantro Static Inputs

#Creating the Field
Field_Yield = 5_000 #lb
Plant_Weight = 1 #lb
Total_Plants = int(Field_Yield/Plant_Weight)
Total_Plants_List = range(1,Total_Plants+1)

#not in use because until harvest
Case_Weight = 20 #lb per case
Bunches_Weight = Case_Weight/Plant_Weight

#Water and Season Characteristics
Water_Irrigation_In = 6 #Inches of water per harvest season
Total_L_Season = 40.46*40.46*(0.0254*Water_Irrigation_In)*1000 # one acre 40.46m2 * 0.348 m of water * 10000 to convert tot m3
Days_per_season = 45 #days
#L_water_day = Total_L_Season/Irrigation_Days_per_season


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

#%%

#B1_Daily testing water (DTW) - Low
Scen_B31_L_DTW = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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

#B1_Daily testing water (DTW) - High
Scen_B31_H_DTW = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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
Scen_B31_L_DTP = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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
Scen_B31_H_DTP = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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

#B1_Water testing 1 time per season at the end of the season - Low
Scen_B31_L_WT1 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [45], #testing water on day 1
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 0, 
                  Product_Testing_H = 0
                  )

#B1_Water testing 1 time per season at the end of the season - High
Scen_B31_H_WT1 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [45], #testing water on day 1
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 0, 
                  Product_Testing_H = 0
                  )

#B1_Water testing 1 time per season at the end of the season - Low
Scen_B31_L_WT4 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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

#B1_Water testing 1 time per season at the end of the season - High
Scen_B31_H_WT4 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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

#B1_Water testing 2 times per season (end and mid) - Low
Scen_B31_L_WT2 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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
#B1_Water testing 2 times per season (end and mid) - High
Scen_B31_H_WT2 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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

#B1_Water testing 3 times per season (end,mid,start) - Low
Scen_B31_L_WT3 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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

#B1_Water testing 3 times per season (end,mid,start) - High
Scen_B31_H_WT3 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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

#B1_product testing 1 time per season at the end of the season - Low
Scen_B31_L_PT1 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [0], #testing water on day 1
                  Testing_Day_Product = [45],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

#B1_product testing 1 time per season at the end of the season - High
Scen_B31_H_PT1 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [45],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

#B1_product testing 1 time per season at the end of the season - Low
Scen_B31_L_PT4 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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

#B1_product testing 1 time per season at the end of the season - High
Scen_B31_H_PT4 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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

#B1_product testing 2 times per season end and mid season - Low
Scen_B31_L_PT2 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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
#B1_product testing 2 times per season end and mid season - High
Scen_B31_H_PT2 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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
Scen_B31_L_PT3 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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
Scen_B31_H_PT3 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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

###############################
Scen_B31_H_DWPT = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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

Scen_B31_L_DWPT =Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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
Scen_B31_H_WPT1 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
                  Testing_Scenario=2,#every day sampling
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [45],
                  Testing_Day_Product = [45],
                  Water_Sampling = 1,
                  Product_Sampling_PH = 1, #now product testing is on
                  Product_Testing_H = 0
                  )

#B1_Water testing 1 time per season at the start of the season - Low
Scen_B31_L_WPT1 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [45], #testing water on day 1
                  Testing_Day_Product = [45],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )


Scen_B31_H_WPT4 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
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

Scen_B31_L_WPT4 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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
#----
#B1_Water testing 1 time per season at the start of the season - High
Scen_B31_H_WPT2 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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
Scen_B31_L_WPT2 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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
Scen_B31_H_WPT3 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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
Scen_B31_L_WPT3 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 31,#Contamination 4 days per season at higher levels
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

##SEcond SCen


#B1_Daily testing water (DTW) - Low
Scen_B32_L_DTW = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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

#B1_Daily testing water (DTW) - High
Scen_B32_H_DTW = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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
Scen_B32_L_DTP = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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
Scen_B32_H_DTP = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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

#B1_Water testing 1 time per season at the end of the season - Low
Scen_B32_L_WT1 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [45], #testing water on day 1
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 0, 
                  Product_Testing_H = 0
                  )

#B1_Water testing 1 time per season at the end of the season - High
Scen_B32_H_WT1 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [45], #testing water on day 1
                  Testing_Day_Product = [0],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 0, 
                  Product_Testing_H = 0
                  )

#B1_Water testing 1 time per season at the end of the season - Low
Scen_B32_L_WT4 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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

#B1_Water testing 1 time per season at the end of the season - High
Scen_B32_H_WT4 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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

#B1_Water testing 2 times per season (end and mid) - Low
Scen_B32_L_WT2 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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
#B1_Water testing 2 times per season (end and mid) - High
Scen_B32_H_WT2 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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

#B1_Water testing 3 times per season (end,mid,start) - Low
Scen_B32_L_WT3 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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

#B1_Water testing 3 times per season (end,mid,start) - High
Scen_B32_H_WT3 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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

#B1_product testing 1 time per season at the end of the season - Low
Scen_B32_L_PT1 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [0], #testing water on day 1
                  Testing_Day_Product = [45],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

#B1_product testing 1 time per season at the end of the season - High
Scen_B32_H_PT1 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [0], 
                  Testing_Day_Product = [45],
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )

#B1_product testing 1 time per season at the end of the season - Low
Scen_B32_L_PT4 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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

#B1_product testing 1 time per season at the end of the season - High
Scen_B32_H_PT4 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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

#B1_product testing 2 times per season end and mid season - Low
Scen_B32_L_PT2 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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
#B1_product testing 2 times per season end and mid season - High
Scen_B32_H_PT2 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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
Scen_B32_L_PT3 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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
Scen_B32_H_PT3 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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

###############################
Scen_B32_H_DWPT = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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

Scen_B32_L_DWPT =Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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
Scen_B32_H_WPT1 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
                  Testing_Scenario=2,#every day sampling
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [45],
                  Testing_Day_Product = [45],
                  Water_Sampling = 1,
                  Product_Sampling_PH = 1, #now product testing is on
                  Product_Testing_H = 0
                  )

#B1_Water testing 1 time per season at the start of the season - Low
Scen_B32_L_WPT1 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = [45], #testing water on day 1
                  Testing_Day_Product = [45],
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 1, 
                  Product_Testing_H = 0
                  )


Scen_B32_H_WPT4 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =20,
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

Scen_B32_L_WPT4 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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
#----
#B1_Water testing 1 time per season at the start of the season - High
Scen_B32_H_WPT2 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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
Scen_B32_L_WPT2 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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
Scen_B32_H_WPT3 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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
Scen_B32_L_WPT3 = Process_Model_Extra(
                  Days_per_season = 45,
                  Niterations= 10000,
                  Custom_Irrigation_Days = 4,
                  Cont_Scenario = 32,#Contamination 4 days per season at higher levels
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

#%% Extracting 
 
    #Baseline 1 - High

get_dec_rate(df  = Scen_B31_H_DTW, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B31_H_WT1, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B31_H_WT4, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B31_H_WT2, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B31_H_WT3, Water_Produce_Both = "Water")


get_dec_rate(df  = Scen_B31_H_DTP, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B31_H_PT1, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B31_H_PT4, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B31_H_PT2, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B31_H_PT3, Water_Produce_Both = "Produce")


get_dec_rate(df  = Scen_B31_H_DWPT, Water_Produce_Both = "Both")
get_dec_rate(df  = Scen_B31_H_WPT1, Water_Produce_Both = "Both")
get_dec_rate(df  = Scen_B31_H_WPT4, Water_Produce_Both = "Both")
get_dec_rate(df  = Scen_B31_H_WPT2, Water_Produce_Both = "Both") 
get_dec_rate(df  = Scen_B31_H_WPT3, Water_Produce_Both = "Both") 


    #Baseline 1 - Low

get_dec_rate(df  = Scen_B31_L_DTW, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B31_L_WT1, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B31_L_WT4, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B31_L_WT2, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B31_L_WT3, Water_Produce_Both = "Water")


get_dec_rate(df  = Scen_B31_L_DTP, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B31_L_PT1, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B31_L_PT4, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B31_L_PT2, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B31_L_PT3, Water_Produce_Both = "Produce")

get_dec_rate(df  = Scen_B31_L_DWPT, Water_Produce_Both = "Both")
get_dec_rate(df  = Scen_B31_L_WPT1, Water_Produce_Both = "Both")
get_dec_rate(df  = Scen_B31_L_WPT4, Water_Produce_Both = "Both")
get_dec_rate(df  = Scen_B31_L_WPT2, Water_Produce_Both = "Both") 
get_dec_rate(df  = Scen_B31_L_WPT3, Water_Produce_Both = "Both") 



###B2

get_dec_rate(df  = Scen_B32_H_DTW, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B32_H_WT1, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B32_H_WT4, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B32_H_WT2, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B32_H_WT3, Water_Produce_Both = "Water")


get_dec_rate(df  = Scen_B32_H_DTP, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B32_H_PT1, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B32_H_PT4, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B32_H_PT2, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B32_H_PT3, Water_Produce_Both = "Produce")


get_dec_rate(df  = Scen_B32_H_DWPT, Water_Produce_Both = "Both")
get_dec_rate(df  = Scen_B32_H_WPT1, Water_Produce_Both = "Both")
get_dec_rate(df  = Scen_B32_H_WPT4, Water_Produce_Both = "Both")
get_dec_rate(df  = Scen_B32_H_WPT2, Water_Produce_Both = "Both") 
get_dec_rate(df  = Scen_B32_H_WPT3, Water_Produce_Both = "Both") 


    #Baseline 1 - Low

get_dec_rate(df  = Scen_B32_L_DTW, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B32_L_WT1, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B32_L_WT4, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B32_L_WT2, Water_Produce_Both = "Water")
get_dec_rate(df  = Scen_B32_L_WT3, Water_Produce_Both = "Water")


get_dec_rate(df  = Scen_B32_L_DTP, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B32_L_PT1, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B32_L_PT4, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B32_L_PT2, Water_Produce_Both = "Produce")
get_dec_rate(df  = Scen_B32_L_PT3, Water_Produce_Both = "Produce")

get_dec_rate(df  = Scen_B32_L_DWPT, Water_Produce_Both = "Both")
get_dec_rate(df  = Scen_B32_L_WPT1, Water_Produce_Both = "Both")
get_dec_rate(df  = Scen_B32_L_WPT4, Water_Produce_Both = "Both")
get_dec_rate(df  = Scen_B32_L_WPT2, Water_Produce_Both = "Both") 
get_dec_rate(df  = Scen_B32_L_WPT3, Water_Produce_Both = "Both") 


#%%

list_of_clusters = []
for i in list(range(11)):
    list_of_clusters.append(100/2**i)

det_rate = []
samp_rate = []
assay_rate = []
for i in list_of_clusters:
    x = Process_Model(Days_per_season = 45,
                      Niterations= 10000,
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
    samp_rate.append(Extracting_outputs_sample_prob(x)[0])
    assay_rate.append(Extracting_outputs_sample_prob(x)[1])

Detection_Rates_High_DPT_List = [item for items in det_rate for item in items]
Detection_Rates_High_DPT = pd.DataFrame({"Drates": Detection_Rates_High_DPT_List,
                                          "Cluster": list_of_clusters,
                                          "samp_rate":samp_rate,
                                          "assay_rate":assay_rate})

Detection_Rates_High_DPT.to_csv("C://Users/gareyes3/Documents/GitHub/CPS-Farm-To-Facility-Cilantro/Detection_Rates_High_DPT_LOC_5k.csv")

#Propduct end of day
det_rate_FPT = []
samp_rate_FTP = []
assay_rate_FTP = []
for i in list_of_clusters :
    x = Process_Model(Days_per_season = 45,
                      Niterations= 10000,
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
    det_rate_FPT.append(get_dec_rate(df= x , Water_Produce_Both= "Produce")) 
    samp_rate_FTP.append(Extracting_outputs_sample_prob(x)[0])
    assay_rate_FTP.append(Extracting_outputs_sample_prob(x)[1])

Detection_Rates_High_FPT_List = [item for items in det_rate_FPT for item in items]
Detection_Rates_High_FPT = pd.DataFrame({"Drates": Detection_Rates_High_FPT_List,
                                          "Cluster": list_of_clusters,
                                          "samp_rate":samp_rate_FTP,
                                          "assay_rate":assay_rate_FTP})

Detection_Rates_High_FPT.to_csv("C://Users/gareyes3/Documents/GitHub/CPS-Farm-To-Facility-Cilantro/Detection_Rates_High_FPT_LOC_5k.csv")

#Propduct end of day
det_rate_DPT_L = []
samp_rate_DPT_L =[]
assay_rate_DPT_L = []
for i in list_of_clusters :
    x = Process_Model(Days_per_season = 45,
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
                      Per_Cont_Field = i,
                      Water_Sampling = 0,
                      Product_Sampling_PH = 1,
                      Product_Testing_H = 0
                      )
    det_rate_DPT_L.append(get_dec_rate(df= x , Water_Produce_Both= "Produce"))  
    samp_rate_DPT_L.append(Extracting_outputs_sample_prob(x)[0])
    assay_rate_DPT_L.append(Extracting_outputs_sample_prob(x)[1])

Detection_Rates_Low_DPT = [item for items in det_rate_DPT_L for item in items]
Detection_Rates_Low_DPT = pd.DataFrame({"Drates": Detection_Rates_Low_DPT,
                                          "Cluster": list_of_clusters,
                                          "samp_rate":samp_rate_DPT_L,
                                          "assay_rate":assay_rate_DPT_L})


Detection_Rates_Low_DPT.to_csv("C://Users/gareyes3/Documents/GitHub/CPS-Farm-To-Facility-Cilantro/Detection_Rates_Low_DPT_LOC_5k.csv")




#Propduct end of day
det_rate_FPT_L = []
samp_rate_FPT_L =[]
assay_rate_FPT_L = []
for i in list_of_clusters :
    x = Process_Model(Days_per_season = 45,
                      Niterations= 10000,
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
                      Per_Cont_Field = i,
                      Water_Sampling = 0,
                      Product_Sampling_PH = 1,
                      Product_Testing_H = 0
                      )
    det_rate_FPT_L.append(get_dec_rate(df= x , Water_Produce_Both= "Produce"))  
    samp_rate_FPT_L.append(Extracting_outputs_sample_prob(x)[0])
    assay_rate_FPT_L.append(Extracting_outputs_sample_prob(x)[1])

Detection_Rates_Low_FPT = [item for items in det_rate_FPT_L for item in items]
Detection_Rates_Low_FPT = pd.DataFrame({"Drates": Detection_Rates_Low_FPT,
                                          "Cluster": list_of_clusters,
                                          "samp_rate":samp_rate_FPT_L,
                                          "assay_rate":assay_rate_FPT_L})


Detection_Rates_Low_FPT.to_csv("C://Users/gareyes3/Documents/GitHub/CPS-Farm-To-Facility-Cilantro/Detection_Rates_Low_FPT_LOC_5k.csv")


#Now Doing 10 samples of 25g

#Propduct end of day
det_rate_FPT_10s_L = []
samp_rate_FPT_10s_L =[]
assay_rate_FPT_10s_L = []
for i in list_of_clusters :
    x = Process_Model(Days_per_season = 45,
                      Niterations= 10000,
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
                      Per_Cont_Field = i,
                      Water_Sampling = 0,
                      Product_Sampling_PH = 1,
                      Product_Testing_H = 0,
                      #sampling
                      N_Samples_Prod = 10,
                      N_Grabs_Prod = 1,
                      )
    det_rate_FPT_10s_L.append(get_dec_rate(df= x , Water_Produce_Both= "Produce"))  
    samp_rate_FPT_10s_L.append(Extracting_outputs_sample_prob(x)[0])
    assay_rate_FPT_10s_L.append(Extracting_outputs_sample_prob(x)[1])

Detection_Rates_Low_10s_FPT = [item for items in det_rate_FPT_10s_L for item in items]
Detection_Rates_Low_10s_FPT = pd.DataFrame({"Drates": Detection_Rates_Low_10s_FPT,
                                          "Cluster": list_of_clusters,
                                          "samp_rate":samp_rate_FPT_10s_L,
                                          "assay_rate":assay_rate_FPT_10s_L})


Detection_Rates_Low_10s_FPT.to_csv("C://Users/gareyes3/Documents/GitHub/CPS-Farm-To-Facility-Cilantro/Detection_Rates_Low_10s_FPT_LOC_5k.csv")


#Now Doing 10 samples of 25g

#Propduct end of day
det_rate_FPT_10s_H = []
samp_rate_FPT_10s_H =[]
assay_rate_FPT_10s_H = []
for i in list_of_clusters :
    x = Process_Model(Days_per_season = 45,
                      Niterations= 10000,
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
                      Product_Testing_H = 0,
                      #sampling
                      N_Samples_Prod = 10,
                      N_Grabs_Prod = 1
                      )
    det_rate_FPT_10s_H.append(get_dec_rate(df= x , Water_Produce_Both= "Produce"))  
    samp_rate_FPT_10s_H.append(Extracting_outputs_sample_prob(x)[0])
    assay_rate_FPT_10s_H.append(Extracting_outputs_sample_prob(x)[1])

Detection_Rates_High_10s_FPT = [item for items in det_rate_FPT_10s_H for item in items]
Detection_Rates_High_10s_FPT = pd.DataFrame({"Drates": Detection_Rates_High_10s_FPT,
                                          "Cluster": list_of_clusters,
                                          "samp_rate":samp_rate_FPT_10s_H,
                                          "assay_rate":assay_rate_FPT_10s_H})


Detection_Rates_High_10s_FPT.to_csv("C://Users/gareyes3/Documents/GitHub/CPS-Farm-To-Facility-Cilantro/Detection_Rates_High_10s_FPT_LOC_5k.csv")



#Now Doing 10 samples of 25g

#Propduct end of day
det_rate_FPT_45s_H = []
samp_rate_FPT_45s_H =[]
assay_rate_FPT_45s_H = []
for i in list_of_clusters :
    x = Process_Model(Days_per_season = 45,
                      Niterations= 10000,
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
                      Product_Testing_H = 0,
                      #sampling
                      N_Samples_Prod = 45,
                      N_Grabs_Prod = 1
                      )
    det_rate_FPT_45s_H.append(get_dec_rate(df= x , Water_Produce_Both= "Produce"))  
    samp_rate_FPT_45s_H.append(Extracting_outputs_sample_prob(x)[0])
    assay_rate_FPT_45s_H.append(Extracting_outputs_sample_prob(x)[1])

Detection_Rates_High_45s_FPT = [item for items in det_rate_FPT_45s_H for item in items]
Detection_Rates_High_45s_FPT = pd.DataFrame({"Drates": Detection_Rates_High_45s_FPT,
                                          "Cluster": list_of_clusters,
                                          "samp_rate":samp_rate_FPT_45s_H,
                                          "assay_rate":assay_rate_FPT_45s_H})


Detection_Rates_High_45s_FPT.to_csv("C://Users/gareyes3/Documents/GitHub/CPS-Farm-To-Facility-Cilantro/Detection_Rates_High_45s_FPT_LOC_5k.csv")



#Now Doing 10 samples of 25g

#Propduct end of day
det_rate_FPT_45s_L = []
samp_rate_FPT_45s_L =[]
assay_rate_FPT_45s_L = []
for i in list_of_clusters :
    x = Process_Model(Days_per_season = 45,
                      Niterations= 10000,
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
                      Per_Cont_Field = i,
                      Water_Sampling = 0,
                      Product_Sampling_PH = 1,
                      Product_Testing_H = 0,
                      #sampling
                      N_Samples_Prod = 45,
                      N_Grabs_Prod = 1
                      )
    det_rate_FPT_45s_L.append(get_dec_rate(df= x , Water_Produce_Both= "Produce"))  
    samp_rate_FPT_45s_L.append(Extracting_outputs_sample_prob(x)[0])
    assay_rate_FPT_45s_L.append(Extracting_outputs_sample_prob(x)[1])

Detection_Rates_Low_45s_FPT = [item for items in det_rate_FPT_45s_L for item in items]
Detection_Rates_Low_45s_FPT = pd.DataFrame({"Drates": Detection_Rates_Low_45s_FPT,
                                          "Cluster": list_of_clusters,
                                          "samp_rate":samp_rate_FPT_45s_L,
                                          "assay_rate":assay_rate_FPT_45s_L})

Detection_Rates_Low_45s_FPT.to_csv("C://Users/gareyes3/Documents/GitHub/CPS-Farm-To-Facility-Cilantro/Detection_Rates_Low_45s_FPT_LOC_5k.csv")
