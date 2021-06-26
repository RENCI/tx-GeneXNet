# collection of functions to read in the data according to a dictionary guide telling which source and outocme to link.
# main driver function is load_pretrain_data
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# this is used for splitting training into train and validation. In particular, the same seed allows us to ensure we are using the same validation data
import random
seedVal=533
random.seed(seedVal)

#load initial data to start by just reading in from h5
def load(path, source):
  print("Loading "+source+" data")
  data={}

  h5file=os.path.join(path, source, source+"_data.h5")
  train_data=pd.read_hdf(h5file, 'train_data')
  test_data=pd.read_hdf(h5file, 'test_data')
  data["train_data"]=train_data
  data["test_data"]=test_data
  data["covariates"]=list(train_data.columns.values)
  print("Done Loading "+source+" data")
  return data

# align outcomes for multi class classification. With multi class outcomes, to make this amenable to NN prediction we must one hot code the data - this is done with the following
def alignOutcomes(ytrain, ytest):
  outcomeData=ytrain.append(ytest)
  trainEndIndex=ytrain.shape[0]

  encoder=LabelEncoder()
  encoder.fit(outcomeData)
  encoded_outcome=encoder.transform(outcomeData)
  dummy_outcome=to_categorical(encoded_outcome)
  return dummy_outcome[0:trainEndIndex,], dummy_outcome[trainEndIndex:dummy_outcome.shape[0],]

# for main data we adjust it to 0/1 labels according to no ASD, ASD respectively
def transformMainOutcome(outcomeData, targets):
  outcomeData[outcomeData=="proband"]=targets[1]
  outcomeData[outcomeData=="TD"]=targets[0]
  outcomeData.columns=['labels']
  return outcomeData

# transform outcomes of data to numerical equivalents
def transformOutcome(outcomeData, outcome, targets):
  if outcome=="DTHHRDY": #https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/variable.cgi?study_id=phs000424.v4.p1&phv=169092
    tau=3.0
    oneIndices= outcomeData >= tau
    oneIndices2= outcomeData==0.
    oneIndices= oneIndices | oneIndices2
    zeroIndices= outcomeData < tau #& outcomeData > 0.

    oneIndices=np.nonzero(np.array(oneIndices))[0]
    zeroIndices=np.nonzero(np.array(zeroIndices))[0]
    if (len(oneIndices) > 0):
      outcomeData.iloc[oneIndices,:]=targets[1]
    if (len(zeroIndices) > 0):
      outcomeData.iloc[zeroIndices,:]=targets[0]
    #0.0 is ventilator - also should be 1
  elif outcome=="sample_type" or outcome=="_sample_type":  #tcga, target respectively
    #https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tcga-study-abbreviations
    outcomeData[outcomeData!="Solid Tissue Normal"]=targets[1]
    outcomeData[outcomeData=="Solid Tissue Normal"]=targets[0]
  elif outcome=="labels":
    outcomeData=transformMainOutcome(outcomeData, targets)
  outcomeData.columns=['labels']
  return outcomeData

# readjust indices for names to be unique to get unique samples
def shaveOffRowNames(data, src):
  if src=="target":
    upToIndex=3
    sample_ids=data["train_data"].index.map(lambda x: '-'.join(x.split('-')[:upToIndex]))
    data["train_data"].set_index(sample_ids, inplace=True)
    data["train_data"]=data["train_data"].drop_duplicates()
    sample_ids=data["test_data"].index.map(lambda x: '-'.join(x.split('-')[:upToIndex]))
    data["test_data"].set_index(sample_ids, inplace=True)
    data["test_data"]=data["test_data"].drop_duplicates()
  elif src=="tcga":
    upToIndex=3
    sample_ids=data["train_data"].index.map(lambda x: '-'.join(x.split('-')[:upToIndex]))
    data["train_data"].set_index(sample_ids, inplace=True)
    data["train_data"]=data["train_data"].drop_duplicates()
    sample_ids=data["test_data"].index.map(lambda x: '-'.join(x.split('-')[:upToIndex]))
    data["test_data"].set_index(sample_ids, inplace=True)
    data["test_data"]=data["test_data"].drop_duplicates()
  return data

def filterRows(data, src):
  samplesToKeep="Primary Solid Tumor"
  if src=="target":
    sampleColumn="_sample_type"
    data["train_data"]=data["train_data"][data["train_data"][sampleColumn].str.contains(samplesToKeep)]
    data["test_data"]=data["test_data"][data["test_data"][sampleColumn].str.contains(samplesToKeep)]
  elif src=="tcga":
    sampleColumn="sample_type"
    data["train_data"]=data["train_data"][data["train_data"][sampleColumn].str.contains(samplesToKeep)]
    data["test_data"]=data["test_data"][data["test_data"][sampleColumn].str.contains(samplesToKeep)]

  # tcga - strip id names to get uqniue samples
  data=shaveOffRowNames(data,src)
  return data

#transfsorm outcomes and get xtrain/xtest data
def reindex(data, keep, outcome, targets, src, multi=False):
  print('adding outcome: '+outcome)
  #if multi, make new data obj
  print(data["train_data"][[outcome]])
  print(type(data["train_data"][outcome]))
  s=set(data["train_data"][outcome].tolist())
  t=set(data["test_data"][outcome].tolist())
  s=s.union(t)
  print(s)
  if multi:
    #prefilter based on outcome - that is, remove samples that correspond to normal or metastatic
    data=filterRows(data, src)

    new_data_obj={}
    new_data_obj["xtrain"]=data["train_data"][keep]
    new_data_obj["xtest"]=data["test_data"][keep]

    if outcome!="labels":
      new_data_obj["ytrain"], new_data_obj["ytest"]=alignOutcomes(data["train_data"][[outcome]], data["test_data"][[outcome]])
      print(new_data_obj["ytrain"].shape)
      print(new_data_obj["ytest"].shape)
    new_data_obj["covariates"]=keep
    new_data_obj["outcome"]=outcome
  
    return new_data_obj

  #if not multi, modify the data obj directly
  data["xtrain"]=data["train_data"][keep]
  data["xtest"]=data["test_data"][keep]

  if outcome!="labels":
    if outcome=='SMTSD':
      #reindex to do simplified smtsd's
      print("remapping smtsd")
      data=remapSMTSD(data)
    data["ytrain"], data["ytest"]=alignOutcomes(data["train_data"][[outcome]], data["test_data"][[outcome]])
    print('outcome: '+outcome)
    print(data["ytrain"].shape)
    print(data["ytest"].shape)
  else:
    print('parsing main')
    data["ytrain"] =transformMainOutcome(data["train_data"][[outcome]], targets)
    data["ytest"] =transformMainOutcome(data["test_data"][[outcome]], targets)
    print(data["ytrain"])

    #get validation data in main training
    valInds=random.sample(range(data["ytrain"].shape[0]), 40)
    data["yval"]=data["ytrain"].iloc[valInds]
    data["xval"]=data["xtrain"].iloc[valInds, :]

    data["ytrain"]=data["ytrain"].drop(index=data["ytrain"].index[valInds])
    data["xtrain"]=data["xtrain"].drop(index=data["xtrain"].index[valInds])

    print("resample on main")
    print(data["xtrain"].shape)
    print(data["ytrain"].shape)
    print(data["xval"].shape)
    print(data["yval"].shape)
    print(data["xtest"].shape)
    print(data["ytest"].shape)
    print("im done on main")

  data["covariates"]=keep
  data["outcome"]=outcome

  del data["train_data"]
  del data["test_data"]

  return data

# SMTSD with all 53 , granular types - e.g. Kidney type 1, Kidney type 2 - is too hard, so take to just region
def remapSMTSD(data):
  tissueType="SMTSD"
  data["train_data"][tissueType]=data["train_data"][tissueType].map(lambda x: '-'.join(x.split('-')[:1])) # keep first organ id
  data["train_data"][tissueType]=data["train_data"][tissueType].map(lambda x: x.rstrip()) # remove spaces at end

  data["test_data"][tissueType]=data["test_data"][tissueType].map(lambda x: '-'.join(x.split('-')[:1])) # keep first organ id
  data["test_data"][tissueType]=data["test_data"][tissueType].map(lambda x: x.rstrip()) # remove spaces at end
  return data

# read in ncbi.brain genes filter list into array - use in set intersects to just train on brain
def getFilter(fileName):
  f=open(fileName,"r")
  lines=f.readlines()
  lines=[line.rstrip() for line in lines]
  return lines

# map disease to tissues
# https://ocg.cancer.gov/programs/target/data-matrix
def getDiseaseToTissueMap():
  diseaseToTissueMap={}
  diseaseToTissueMap["AML"]="Whole Blood" # blood/bone marrow
  diseaseToTissueMap["Acute Myeloid Leukemia, Induction Failure Subproject"]="Whole Blood" # blood/bone marrow
  diseaseToTissueMap["Acute Lymphoblastic Leukemia"]="Whole Blood" # cancer of blood/bone marrow - starts in bone marrow
  diseaseToTissueMap["Neuroblastoma"]="Nerve" 
  diseaseToTissueMap["Wilms Tumor"]="Kidney"
  diseaseToTissueMap["Clear Cell Sarcoma of the Kidney"]="Kidney"
  diseaseToTissueMap["Kidney, Rhabdoid Tumor"]="Kidney"
  return diseaseToTissueMap

# map between tissue sources using the above dictionary
def mapDiseaseToTissue(diseaseTrain, diseaseTest):
  diseaseToTissueMap=getDiseaseToTissueMap()
  diseaseTrain=diseaseTrain.map(lambda x: diseaseToTissueMap[x]) 
  diseaseTest=diseaseTest.map(lambda x: diseaseToTissueMap[x]) 
  return diseaseTrain, diseaseTest

#https://stephenturner.github.io/tcga-codes/
#https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga/studied-cancers
def getCancerToTissueMap():
  cancerToTissueMap={}
  cancerToTissueMap["ACC"]="Adrenal Gland"
  cancerToTissueMap["BLCA"]="Bladder"
  cancerToTissueMap["BRCA"]="Breast"
  cancerToTissueMap["CESC"]="Cervix"
  cancerToTissueMap["CHOL"]="Bile Duct" #Cholangiocarcinoma: liver, gallbladder, small intestine
  cancerToTissueMap["COAD"]="Colon"
  cancerToTissueMap["COADREAD"]="Colon" #Colorectal adenocarcinoma
  cancerToTissueMap["DLBC"]="Lymph Node" #Lymphoid Neoplasm Diffuse Large B-cell Lymphoma - any lymph nodes?
  cancerToTissueMap["ESCA"]="Esophagus"
  cancerToTissueMap["FPPP"]="FPPP" # FFPE Pilot Phase II - formalin-fixed paraffin-embedded ??
  cancerToTissueMap["GBM"]="Brain"
  cancerToTissueMap["GBMLGG"]="Brain"
  cancerToTissueMap["HNSC"]="HeadNeck" #head/neck squamous cell carcinoma - maybe like skin?
  cancerToTissueMap["KICH"]="Kidney"
  cancerToTissueMap["KIPAN"]="Kidney"
  cancerToTissueMap["KIRC"]="Kidney"
  cancerToTissueMap["KIRP"]="Kidney"
  cancerToTissueMap["LAML"]="Whole Blood" #acute myeloid leukemia - blood/bone marrow
  cancerToTissueMap["LGG"]="Brain"
  cancerToTissueMap["LIHC"]="Liver"
  cancerToTissueMap["LUAD"]="Lung"
  cancerToTissueMap["LUSC"]="Lung"
  cancerToTissueMap["MESO"]="Mesothelioma" #Mesothelioma - pleura in lung/chest cavity, otherwise in heart or abdominal - maybe like atrial/aorta?/Artery
  cancerToTissueMap["OV"]="Ovary" 
  cancerToTissueMap["PAAD"]="Pancreas" 
  cancerToTissueMap["PCPG"]="Adrenal Gland" #Pheochromocytoma and Paraganglioma - Adrenal Gland, outside of adrenal via Nerve Tissue
  cancerToTissueMap["PRAD"]="Prostate"
  cancerToTissueMap["READ"]="Rectum" # rectum is not in GTEX SMTSD?
  cancerToTissueMap["SARC"]="Bone" # sarcoma # soft tissue - may be very hard to classify - throw out misclassified if wonky - Adipose? Artery?
  cancerToTissueMap["SKCM"]="Skin"
  cancerToTissueMap["STAD"]="Stomach"
  cancerToTissueMap["TGCT"]="Testis"
  cancerToTissueMap["THCA"]="Thyroid"
  cancerToTissueMap["THYM"]="Thymus" # thymus is not in GTEX SMTSD?
  cancerToTissueMap["UCEC"]="Uterus"
  cancerToTissueMap["UCS"]="Uterus"
  cancerToTissueMap["UVM"]="Eye" #eye is not in GTEX SMTSD?
  return cancerToTissueMap

# map between tissue sources using the above dictionary
def mapCancerToTissue(cancerTrain, cancerTest):
  cancerToTissueMap=getCancerToTissueMap()
  cancerTrain=cancerTrain.map(lambda x: cancerToTissueMap[x]) 
  cancerTest=cancerTest.map(lambda x: cancerToTissueMap[x]) 
  return cancerTrain, cancerTest

# when preparing pooled data, we want to collect all the outcomes of interest. In doing this though, we need to pad data or fill in subsitutes for certain outcomes across datasets
# e.g.) hardy death scale classificaiton is in GTEX but not others. We can 'pad' this outcome by including an additional category for TCGA/TARGET samples.
def padOutcomeInPooledData(all_data, source):
  data=all_data[source]
  trainSize, testSize = data["train_data"].shape[0], data["test_data"].shape[0]

  #sampleTrain, sampleTest, hardyTrain, hardyTest, cancerTrain, cancerTest = [], [], [], [], [], []
  tissueTrain, tissueTest, hardyTrain, hardyTest, cancerTrain, cancerTest = [], [], [], [], [], []

  if source=="gtex":
    #gtexSampleEntry="Solid Tissue Normal"
    #sampleTrain, sampleTest = pd.Series([gtexSampleEntry]*trainSize, data["train_data"].index), pd.Series([gtexSampleEntry]*testSize, data["test_data"].index)
    #tissue = "SMTSD"
    #tissueyTrain, tissueyTest = data["train_data"][sample], data["test_data"][sample]
    #one hot encode here 
    tissueType="SMTSD"
    tissueTrain=data["train_data"][tissueType].map(lambda x: '-'.join(x.split('-')[:1])) # keep first organ id
    tissueTrain=tissueTrain.map(lambda x: x.rstrip()) # keep first organ id

    tissueTest=data["test_data"][tissueType].map(lambda x: '-'.join(x.split('-')[:1])) # keep first organ id
    tissueTest=tissueTest.map(lambda x: x.rstrip()) # keep first organ id

    hardy="DTHHRDY"
    hardyTrain, hardyTest = data["train_data"][hardy], data["test_data"][hardy]
    #one hot encode here 

    #pad rest with data
    gtexCancerousEntry=0
    cancerTrain, cancerTest = pd.Series([gtexCancerousEntry]*trainSize, data["train_data"].index), pd.Series([gtexCancerousEntry]*testSize, data["test_data"].index)

  elif source == "tcga":
    #remove tcga sample types that are not favorable here
    sample = "sample_type" #Primary Solid Tumor, Solid Tissue Normal
    trainStr="train_data"
    data["train_data"]=data[trainStr].loc[(data[trainStr][sample] != "Metastatic") & (data[trainStr][sample] != "Additional Metastatic") & (data[trainStr][sample] != "Recurrent Solid Tumor") & (data[trainStr][sample] != "Additional - New Primary")]
    testStr="test_data"
    data["test_data"]=data[testStr].loc[(data[testStr][sample] != "Metastatic") & (data[testStr][sample] != "Additional Metastatic") & (data[testStr][sample] != "Recurrent Solid Tumor") & (data[testStr][sample] != "Additional - New Primary")]
    trainSize, testSize = data["train_data"].shape[0], data["test_data"].shape[0]

    cancerId = "cancer_type"
    tissueTrain, tissueTest = data["train_data"][cancerId], data["test_data"][cancerId]
    tissueTrain, tissueTest = mapCancerToTissue(tissueTrain, tissueTest)

    sample = "sample_type" #Primary Solid Tumor, Solid Tissue Normal
    sampleTrain, sampleTest = data["train_data"][sample], data["test_data"][sample]
    normalTissue="Solid Tissue Normal"
    indicesTrain=sampleTrain==normalTissue
    indicesTest=sampleTest==normalTissue

    #6 on Hardy
    hardyEntry=6
    hardyTrain, hardyTest = pd.Series([hardyEntry]*trainSize, data["train_data"].index), pd.Series([hardyEntry]*testSize, data["test_data"].index)

    gtexCancerousEntry=1
    cancerTrain, cancerTest = pd.Series([gtexCancerousEntry]*trainSize, data["train_data"].index), pd.Series([gtexCancerousEntry]*testSize, data["test_data"].index)
    cancerTrain[indicesTrain]=0.
    cancerTest[indicesTest]=0.
    #set below
    #0 if not cancerous

  elif source=="target":
    diseaseType = "_primary_disease"
    tissueTrain, tissueTest = data["train_data"][diseaseType], data["test_data"][diseaseType]
    tissueTrain, tissueTest = mapDiseaseToTissue(tissueTrain, tissueTest)

    sample="_sample_type" #Primary Solid Tumor,...
    sampleTrain, sampleTest = data["train_data"][sample], data["test_data"][sample]

    normalTissue="Solid Tissue Normal"
    indicesTrain=sampleTrain==normalTissue
    indicesTest=sampleTest==normalTissue

    #6 on Hardy
    hardyEntry=6
    hardyTrain, hardyTest = pd.Series([hardyEntry]*trainSize, data["train_data"].index), pd.Series([hardyEntry]*testSize, data["test_data"].index)

    gtexCancerousEntry=1
    cancerTrain, cancerTest = pd.Series([gtexCancerousEntry]*trainSize, data["train_data"].index), pd.Series([gtexCancerousEntry]*testSize, data["test_data"].index)
    # get indices of sample type == "Solid Tisse Normal"
    cancerTrain[indicesTrain]=0.
    cancerTest[indicesTest]=0.

  return tissueTrain, tissueTest, hardyTrain, hardyTest, cancerTrain, cancerTest

# actually get the outputs from each source in dataset when pooling, and append together.
def mapPoolOutcome(all_data, sourceOrder, outcome_map):
  #return order if just appended will line up, as order is provided in sourceOrder array
  #row is a given sample's outcome, outcome is multi d. |tissue type|+1 (cancerous/not) + 6 (5 from Hardy + 1 for other data in TARGET/TCGA) - though this is subject to change

  print("loading source "+sourceOrder[0])
  sampleTrain, sampleTest, hardyTrain, hardyTest, cancerTrain, cancerTest = padOutcomeInPooledData(all_data, sourceOrder[0])
  print(sampleTrain.value_counts())
  print(sampleTest.value_counts())
  s1=np.unique(sampleTrain)
  s2=np.unique(sampleTrain)
  s=np.intersect1d(s1,s2)
  alls=list(s)
  # get outcome data for each ousrce - append together
  for i in range(1, len(sourceOrder)):
    print("loading source "+sourceOrder[i])
    source=sourceOrder[i]
    #pad this data according to source
    IsampleTrain, IsampleTest, IhardyTrain, IhardyTest, IcancerTrain, IcancerTest = padOutcomeInPooledData(all_data, source)
    print(IsampleTrain.value_counts())
    print(IsampleTest.value_counts())
    s1=np.unique(IsampleTrain)
    s2=np.unique(IsampleTest)
    s=np.intersect1d(s1,s2)
    alls.extend(s)
    #stack outcomes
    sampleTrain=sampleTrain.append(IsampleTrain)
    sampleTest=sampleTest.append(IsampleTest)
    hardyTrain=hardyTrain.append(IhardyTrain)
    hardyTest=hardyTest.append(IhardyTest)
    cancerTrain=cancerTrain.append(IcancerTrain)
    cancerTest=cancerTest.append(IcancerTest)

  #realign the outcomes here - alignoutcomes on hardy, sample, cancer - need to do this after collecting from all sources so that we do not miss any classes in test but not in train, or vice versa
  print("\nAll sample types\n")
  print(set(alls))

  # now align it
  print("\nOutcome dims for each type\n")
  #sampleTrain, sampleTest = alignOutcomes(sampleTrain, sampleTest)
  print("Sample")
  #print(sampleTrain.value_counts())
  #print(sampleTest.value_counts())
  sampleTrain, sampleTest = alignOutcomes(sampleTrain, sampleTest)
  print(sampleTrain.shape)
  print(sampleTest.shape)
  
  print("Hardy")
  #print(hardyTrain.value_counts())
  #print(hardyTest.value_counts())
  hardyTrain, hardyTest = alignOutcomes(hardyTrain, hardyTest)
  print(hardyTrain.shape)
  print(hardyTest.shape)
  #cancerTrain, cancerTest = alignOutcomes(cancerTrain, cancerTest)
  print("Cancer")
  #print(cancerTrain.value_counts())
  #print(cancerTest.value_counts())
  print(cancerTrain.shape)
  print(cancerTest.shape)

  #merge outcome data together
  ytrain=sampleTrain
  #ytrain=np.concatenate([ytrain, hardyTrain], axis=1)
  cancerTrain=np.array(cancerTrain)
  cancerTrain=cancerTrain.reshape((cancerTrain.shape[0], 1))
  ytrain=np.concatenate([ytrain, cancerTrain], axis=1)
  print(ytrain.shape)

  ytest=sampleTest
  #ytest=np.concatenate([ytest, hardyTest], axis=1)
  cancerTest=np.array(cancerTest)
  cancerTest=cancerTest.reshape((cancerTest.shape[0], 1))
  ytest=np.concatenate([ytest, cancerTest], axis=1)

  return ytrain, ytest

# pooled data function driver
def pooledData(all_data, colsToKeep, outcome_map):
  print("\nPreparing pooled data...\n")
  #rbind all data together
  #We pretrain on a neural network using all data sources pooled together as a single source. The network is trained on three objectives: Hardy scale (DTHHRDY), cancerous or not, and sample type. Data from TCGA is encoded as ‘5’ on the Hardy scale, and data from GTEX is labeled as not cancerous.
  data={}

  # start with the first src: append training data togther one ata time from each source
  # remove undesirable samples in tcga: see below under if src=="tcga"
  sources=list(all_data.keys())
  sources.remove("main")
  src=sources[0]
  print("first source is "+src)
  if src=="tcga":
    sample = "sample_type"
    trainStr="train_data"
    all_data[src]["train_data"]=all_data[src][trainStr].loc[(all_data[src][trainStr][sample] != "Metastatic") & (all_data[src][trainStr][sample] != "Additional Metastatic") & (all_data[src][trainStr][sample] != "Recurrent Solid Tumor") & (all_data[src][trainStr][sample] != "Additional - New Primary")]
    testStr="test_data"
    all_data[src]["test_data"]=all_data[src][testStr].loc[(all_data[src][testStr][sample] != "Metastatic") & (all_data[src][testStr][sample] != "Additional Metastatic") & (all_data[src][testStr][sample] != "Recurrent Solid Tumor") & (all_data[src][testStr][sample] != "Additional - New Primary")]
  allDatTrain=all_data[src]["train_data"][colsToKeep]
  allDatTest=all_data[src]["test_data"][colsToKeep]
  print(allDatTrain.shape)
  print(allDatTest.shape)
  sources.remove(src)
  firstSource=src  
  #hardy, cancerous, sample type

  for src in sources:
    if src=="tcga":
      sample = "sample_type" #Primary Solid Tumor, Solid Tissue Normal
      trainStr="train_data"
      all_data[src]["train_data"]=all_data[src][trainStr].loc[(all_data[src][trainStr][sample] != "Metastatic") & (all_data[src][trainStr][sample] != "Additional Metastatic") & (all_data[src][trainStr][sample] != "Recurrent Solid Tumor") & (all_data[src][trainStr][sample] != "Additional - New Primary")]
      testStr="test_data"
      all_data[src]["test_data"]=all_data[src][testStr].loc[(all_data[src][testStr][sample] != "Metastatic") & (all_data[src][testStr][sample] != "Additional Metastatic") & (all_data[src][testStr][sample] != "Recurrent Solid Tumor") & (all_data[src][testStr][sample] != "Additional - New Primary")]

    allDatTrain=allDatTrain.append(all_data[src]["train_data"][colsToKeep])
    allDatTest=allDatTest.append(all_data[src]["test_data"][colsToKeep])
    print(allDatTrain.shape)
    print(allDatTest.shape)

  data["xtrain"], data["xtest"]=allDatTrain, allDatTest

  sources.insert(0, firstSource)
  # get the outputs
  data["ytrain"], data["ytest"]=mapPoolOutcome(all_data, sources, outcome_map)

  print(data["ytrain"])
  print(data["ytest"])
  print(data["ytrain"].shape)
  print(data["ytest"].shape)

  data["covariates"] = colsToKeep
  data["outcome"]="pooledOutcomes"

  all_data["pooled"]=data
  #print("Done loading pooled data")
  return all_data

# load pretrain data
# first load in the data sources
# then get common covariate set (overlapping genes)
# then setup pooled data
# finally set up the source outcomes pairs (classes) for training
# to do: 
#   * remove Target data code, multi
#   * rename 'outcome' to 'class'
def load_pretrain_data(path, source_outcomes, targets=[0, 1]):
  all_data={}
  sources=list(source_outcomes.keys())

  multiMetaInfo=None
  if 'multi' in sources:
    multiMetaInfo=source_outcomes['multi'] #a whole array
    sources.remove('multi')

  #load all data 
  for source in sources:
    data=load(path, source)
    all_data[source]=data

  #get col intersect
  colsToKeep=None
  for source in sources:
    cols=set(all_data[source]["covariates"])
    if colsToKeep==None:
      colsToKeep=cols
    else:
      colsToKeep=colsToKeep.intersection(cols)
  colsToKeep=list(colsToKeep)

  all_data=pooledData(all_data, colsToKeep, source_outcomes)

  #multi is used to say we have multiple outcome maps to a single source. Hence we cannot modify the src object directly as we do in non multi case, or regular case below. 
  # instead, we instantiate a new data obj dictionary and we use that. multi=True denotes make a new dict and don't modify the current standing source data dictionary
  if multiMetaInfo != None:
    keys=list(multiMetaInfo.keys())
    for src in keys:
      outcome=multiMetaInfo[src]
      print('Multi: src/outcome:'+src+'/'+outcome)
      all_data["multi-"+src]=reindex(all_data[src], colsToKeep, outcome, targets, src, multi=True)
      print(all_data["multi-"+src]["ytrain"].shape)

  #now subindex or set data accordingly
  for source in sources:
    print('Regular: src/outcome:'+source+'/'+source_outcomes[source])
    all_data[source]=reindex(all_data[source], colsToKeep, source_outcomes[source], targets, source)
    print(all_data[source]["ytrain"].shape)

  return all_data, source_outcomes

