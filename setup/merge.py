# usage: python merge.py --working_dir ${PRETRAIN_ROOT} --data_dir ${DATA_ROOT} --dry_run
# Called from: setup.sh
# Purpose: Create the labeled h5 files from the tsv + labels files, find the common genes and merge into a single csv
# Inputs: public and private gene expression files + labels, configured via setup.sh and found in:
#   <working_dir>/<source>/<file>,
#     where <source>/<file> are each of:
#        gtex/test_gtex_expression_matrix_processed.tsv.gz
#        gtex/train_gtex_expression_matrix_processed.tsv.gz
#        gtex/GTEx_v7_Annotations_SampleAttributesDS.txt
#        gtex/GTEx_v7_Annotations_SubjectPhenotypesDS.txt
#        main/microarrayASD_Test.csv
#        main/microarrayASD_Train.csv
#        main/microarrayASDLabels_Test.csv
#        main/microarrayASDLabels_Train.csv
#        tcga/test_tcga_expression_matrix_processed.tsv.gz
#        tcga/train_tcga_expression_matrix_processed.tsv.gz
#        tcga/tcga_sample_identifiers.tsv
# Outputs: 
#   <working_dir>/<source>/<source>_data.h5,
#      where <source> is each of: gtex, tcga, main
#   <data_dir>/<source>-<class-type>-<subkey>.csv,
#      where <source>-<class-type> is each of: gtex-SMTSD, tcga-sample_type, main-labels
#      and <subkey> is each of: xtest, xtrain, ytest, ytrain, xval, yval

import os
import pandas as pd
import argparse
import time
import zipfile

parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('-w', '--working_dir', type=str, required=True, help='input and output dir of the files containing data to be processed')
parser.add_argument('-d', '--dry_run', action='store_true', default=False, help='dry run without actually writing any data')
parser.add_argument('-v', '--verbose', action='store_true', default=False, help='print extra messages')
parser.add_argument('-dataDir', '--data_dir', action='store', type=str)
args = parser.parse_args()
working_dir = args.working_dir
debug = args.dry_run
verbose = args.verbose
dataDir = args.data_dir
g_start_time = time.time()
def t_str():
  global g_start_time
  elapsed_time = time.time() - g_start_time
  return "["+str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))+"]"
#---------------------------------
###
if verbose == True:
  print(t_str()+" >>> Create tcga_data.h5 with test_data and train_data keys <<< ")
###
def tcga_csv2h5(samp, path, h5fn, data_type, debug, verbose):
  infile=os.path.join(path, data_type + "_tcga_expression_matrix_processed.tsv.gz")
  if verbose == True:
    print(t_str()+" reading ["+infile+"]... ")
  exprData=pd.read_csv(infile, sep='\t', index_col=0)
  if verbose == True:
    print(t_str()+" merge... ")
  exprData=exprData.merge(samp, how='left', left_index=True, right_index=True)
  if verbose == True:
    print(t_str()+" dropna... ")
  exprData=exprData.dropna()
  if debug == True:
    print("+ exprData.to_hdf("+h5fn+", key="+data_type+"_data))")
  else:
    exprData.to_hdf(h5fn, key=(data_type + '_data'))

tcga_path=os.path.join(working_dir, "tcga")
h5filename=os.path.join(tcga_path, "tcga_data.h5")
if verbose == True:
  print(t_str()+" read sample info... ")
sample_info = pd.read_csv(os.path.join(tcga_path, "tcga_sample_identifiers.tsv"), sep='\t', index_col=0)
if verbose == True:
  print(t_str()+" read train data... ")
tcga_csv2h5(sample_info, tcga_path, h5filename, "train", debug, verbose)
if verbose == True:
  print(t_str()+" read test data... ")
tcga_csv2h5(sample_info, tcga_path, h5filename, "test", debug, verbose)
#---------------------------------
###
if verbose == True:
  print(t_str()+" >>> Create gtex_data.h5 with test_data and train_data keys <<<")
###
def gtex_csv2h5(samp, subj,path, h5fn, data_type, debug, verbose):
  infile=os.path.join(path, data_type + "_gtex_expression_matrix_processed.tsv.gz")
  if verbose == True:
    print(t_str()+" reading ["+infile+"]... ")
  exprData=pd.read_csv(infile, sep='\t', index_col=0)
  #
  samp=samp[samp['SMAFRZE']=='RNASEQ'] #keep the RNASEQ data - other 
  sampleColsToKeep=['SMTSD', 'SMAFRZE'] #additionally maintain tissue type
  samp=samp[sampleColsToKeep]
  if verbose == True:
    print(t_str()+" merge... ")
  exprData=exprData.merge(samp, how='left', left_index=True, right_index=True)
  if verbose == True:
    print(t_str()+" dropna... ")
  exprData=exprData.dropna()
  if verbose == True:
    print(t_str()+" sample_ids... ")
  sample_ids=exprData.index.map(lambda x: '-'.join(x.split('-')[:2])) # parse down id to merge appropriately
  exprData.set_index(sample_ids, inplace=True)
  if verbose == True:
    print(t_str()+" 2nd merge... ")
  exprData=exprData.merge(subj, how='left', left_index=True, right_index=True)
  if verbose == True:
    print(t_str()+" 2nd dropna... ")
  exprData=exprData.dropna()
  #
  if debug == True:
    print("+ exprData.to_hdf("+h5fn+", key="+(data_type+'_data')+"")
  else:
    exprData.to_hdf(h5fn, key=(data_type + '_data'))
gtex_path=os.path.join(working_dir, "gtex")
h5filename=os.path.join(gtex_path, "gtex_data.h5")
if verbose == True:
  print(t_str()+" read sample attributes... ")
sample_info=pd.read_csv(os.path.join(gtex_path, "GTEx_v7_Annotations_SampleAttributesDS.txt"), sep='\t', index_col=0)
if verbose == True:
  print(t_str()+" read subject phenotypes... ")
subject_info=pd.read_csv(os.path.join(gtex_path, "GTEx_v7_Annotations_SubjectPhenotypesDS.txt"), sep='\t', index_col=0)
if verbose == True:
  print(t_str()+" read train data... ")
gtex_csv2h5(sample_info, subject_info, gtex_path, h5filename, "train", debug, verbose)
if verbose == True:
  print(t_str()+" read test data... ")
gtex_csv2h5(sample_info, subject_info, gtex_path, h5filename, "test", debug, verbose)
#---------------------------------
###
if verbose == True:
  print(t_str()+" >>> Create main_data.h5 with test_data and train_data keys <<< ")
###
def main_csv2h5(path, infile, h5fn, data_type, debug):
  exprData = pd.read_csv(os.path.join(path, infile), sep=',', index_col=0)
  if debug == True:
    print("+ exprData.to_hdf("+ h5fn + ", key="+data_type+"_data)")
  else:
    exprData.to_hdf(h5fn, key=(data_type+'_data'))

main_path=os.path.join(working_dir, "main")
h5filename = os.path.join(main_path, "main_data.h5")
if verbose == True:
  print(t_str()+" read train data... ")
main_csv2h5(main_path, "microarrayASD_Train.csv", h5filename, "train", debug)
if verbose == True:
  print(t_str()+" read test data... ")
main_csv2h5(main_path, "microarrayASD_Test.csv", h5filename, "test", debug)

#----------------------------------
###
if verbose == True:
  print(t_str()+" >>> Read all the labeled h5's and output to a single, 1-hot-encoded csv that only retains common genes <<< ")
###
import numpy as np

# targets for ASD binary or binary problems - cannot use factor levels or strings, need numerical representation
targets=[0,1]
# the map defining what outcome to map each src data to, and if there are more than one data object to prepare in that manner ('see tcga')
source_outcomes={'tcga': 'sample_type', 'gtex': 'SMTSD', 'main': 'labels', 'multi': {'tcga': 'cancer_type'}}
# to do: these are actually classes, not outcomes; source_outcomes => source_classes
# to do: omit multi-tcga-cancer_type-ytest.csv, pooled-pooledOutcomes-ytest.csv?

# load the data
from read_functions import load_pretrain_data
# read each h5 created above and save only common genes
if debug == True:
  print("+ all_data, source_outcomes=load_pretrain_data("+working_dir+", source_outcomes, targets)")
  exit()
else:
  # to do: refactor read_functions
  all_data, source_outcomes=load_pretrain_data(working_dir, source_outcomes, targets)

# to do: create all_data incrementally from steps above?

keys=list(all_data.keys())
print("sources "+str(keys))

if not os.path.exists(dataDir):
  os.makedirs(dataDir)

toRm=['covariates', 'outcome']
for key in list(all_data.keys()):
  print(all_data[key].keys())
  print(key +" "+ all_data[key]['outcome'] +" - ytrain :")
  outcome=all_data[key]['outcome']
  subkeys=list(all_data[key].keys())
  subkeys=list(set(subkeys).difference(toRm))

  for subkey in subkeys:
    path=os.path.join(dataDir, key+"-"+outcome+"-"+subkey+".csv")
    if type(all_data[key][subkey])==type(np.ndarray([0])):
      #convert and write
      df = pd.DataFrame(all_data[key][subkey])
    else:
      df = all_data[key][subkey]
    gene_names=sorted(df.columns)
    if debug == True:
      print("+ df.to_csv(path, index=False, )")
    else:
      df.to_csv(path, index=False,  columns=gene_names)
