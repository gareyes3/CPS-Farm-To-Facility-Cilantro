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
Per_Cont_Field = 100

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
                  #Contamination Information
                  OO_per_L,
                  #Water Testing Options
                  Sampling_every_Days_Water, #change to 1 always
                  Sampling_every_Days_Product, #change to 1 always
                  #for scenario 2
                  Testing_Day_Water, #if scenario 2, then select the days you want testing to happen 1,23,45
                  Testing_Day_Product,#if scenario 2, then select the days you want testing to happen 1,23,45
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
        
        #Contmaination scenario selection
        if Cont_Scenario ==2:
            Random_Irr_Day_Scen2 = random.randint(1,Days_per_season)
                
        #Water Sampling
        #Sampling_every_Days_Water = 1
        if Testing_Scenario ==1 :
            Water_Sampling_Days = np.arange(1,Days_per_season+1,Sampling_every_Days_Water)
        elif Testing_Scenario ==2:
            Water_Sampling_Days = [Testing_Day_Water]
        
        #Product Sampling Days
        #Sampling_every_Days_Product = 1
        if Testing_Scenario ==1 :
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


    
 #%% SCENARIOS     THESE ARE EXAMPLES (SOME REAL SCENARIO), TUNE THEM TO MATCH THE SCENARIOS WE SPOKE ABOUT  IN DOC
#FOR THE FINAL ANALYSIS nITERATIONS ==10,000  
       
#Baseline Scenario 1, No Sampling one a day. Contamination every Day
#Low Contamination
Out_B1_L = Process_Model(
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
Out_B1_H = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10,
                  Cont_Scenario = 1,#Change in baseline scenario
                  Testing_Scenario=2,#leave as 2 default
                  #Contamination Information
                  OO_per_L =20,
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

#Baseline Scenario 2: Irrigation one day randomly
Out_B2_L = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10,
                  Cont_Scenario = 2, #Change in baseline scenario
                  Testing_Scenario=2, #leave as 2 default
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

Out_B2_H = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10,
                  Cont_Scenario = 2, #Change in baseline scenario
                  Testing_Scenario=2, #leave as 2 default
                  #Contamination Information
                  OO_per_L =20,
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



#BaselineScenario 1: Plus sampling plans =========================================
#B1_Daily testing water (DTW) -Low
Scen_B1_L_DTW = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=1,#every day sampling
                  #Contamination Information
                  OO_per_L =0.6,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = 0,
                  Testing_Day_Product = 0,
                  Water_Sampling = 1,
                  Product_Sampling_PH = 0,
                  Product_Testing_H = 0
                  )

Scen_B1_H_DTW = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=1,#every day sampling
                  #Contamination Information
                  OO_per_L =20,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as fault 
                  #Testing Options
                  Testing_Day_Water = 0,
                  Testing_Day_Product = 0,
                  Water_Sampling = 1,
                  Product_Sampling_PH = 0,
                  Product_Testing_H = 0
                  )


#este ejemplo es de product testing todos los dias: 
Scen_B1_L_DTWP = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=1,#every day sampling
                  #Contamination Information
                  OO_per_L =0.06,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #1 for sampling every day
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = 0,
                  Testing_Day_Product = 0,
                  Water_Sampling = 0,
                  Product_Sampling_PH = 1, #now product testing is on
                  Product_Testing_H = 0
                  )


#este ejemplo es el de water testing en el dia 1
Scen_B1_L_WT1 = Process_Model(
                  Days_per_season = 45,
                  Niterations= 10,
                  Cont_Scenario = 1,#every day contamiantion
                  Testing_Scenario=2,#Testing only in given day
                  #Contamination Information
                  OO_per_L =0.06,
                  #Water Testing Options
                  Sampling_every_Days_Water = 1, #leave as 1 defaul
                  Sampling_every_Days_Product = 1, #as default 
                  #Testing Options
                  Testing_Day_Water = 1, #testing water on day 1
                  Testing_Day_Product = 0,
                  Water_Sampling = 1,#now water testing is on
                  Product_Sampling_PH = 0, 
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

#THIS GIVES YOU THE ENDPOINT TAC, ADJUST ACCORDINGLY PER SCENARIO

#This should get the exposure of any of the scenarios: just replace the dataframe. at the end this can be automated. as well
get_exposure(df_rejections =Scen_B1_L_DTW[0] ,df_FinalConts = Scen_B1_L_DTW[8],Iterations = 10,Days_per_season=45) #el 0 es para el water esting

#this gets you the output TAC  of scenario with product testing
get_exposure(df_rejections =Scen_B1_L_DTWP[2] ,df_FinalConts = Scen_B1_L_DTWP[8],Iterations = 10, Days_per_season=45) #el 2 es para el product esting

#enpoint TAC for testing only on day 1 water
get_exposure(df_rejections =Scen_B1_L_WT1[0] ,df_FinalConts = Scen_B1_L_WT1[8],Iterations = 10, Days_per_season=45) #el 2 es para el product esting





#
get_exposure_WP(df_rejections_W = Scen_B1_L_DTWP[0],df_rejections_P = Scen_B1_L_DTWP[2],df_FinalConts = Scen_B1_L_DTWP[8],Iterations =10, Days_per_season =45)
#%% Nothing yet, trying code
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
    

