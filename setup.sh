#!/bin/bash
# usage: ./setup.sh
# Purpose: Retrieve and format training/testing data
# 
# Exported environmental variables:
#   RAW_DATA_ROOT - location to store raw data, preferably above the git repo root
#   MODEL_ROOT - location for storing interim, pre-trained models
#   SOURCE_MAIN_DATA_ROOT - location of original, source data to use for training the 'main' model (e.g., ASD)
# 
#   PRETRAIN_PATH - path under the MODEL_ROOT into which the pretrain data will go
#   SOURCE_MAIN_DATA_RDA - path under SOURCE_MAIN_DATA_ROOT for finding 'data.rda'
# 
# Outputs:
#   ${RAW_DATA_ROOT}/* - Original, raw data
#   ${MODEL_ROOT}/*    - Directory structure and data for training
#
# Benchmark: ~15min


##
# 1. Initialize environment
##
pip install -r requirements.txt
mkdir -p ${RAW_DATA_ROOT}

##
# 2. Retrieve public data and main model data for pre-training
##
pushd ${RAW_DATA_ROOT}
## main model data (see manuscript)
cp ${SOURCE_MAIN_DATA_ROOT}/inputDataJabba_main.rda .
cp ${SOURCE_MAIN_DATA_ROOT}/${SOURCE_MAIN_DATA_RDA}/data.rda .
# Retrieve version-locked, pre-formatted TCGA and GTEX data from the BioBombe project:
## tcga
wget https://raw.githubusercontent.com/greenelab/BioBombe/3f9500acff2fb96b41152cdba8e3016cb5fc1bd0/0.expression-download/data/tcga_sample_identifiers.tsv
wget https://github.com/greenelab/BioBombe/raw/5476ce29cb38e8bb1362528c5b0d24d271664680/0.expression-download/data/test_tcga_expression_matrix_processed.tsv.gz
wget https://github.com/greenelab/BioBombe/raw/5476ce29cb38e8bb1362528c5b0d24d271664680/0.expression-download/data/train_tcga_expression_matrix_processed.tsv.gz
## gtex
wget https://github.com/greenelab/BioBombe/raw/f0ce9d2d91849a703b5da297a837b105c7c7b4fa/0.expression-download/data/test_gtex_expression_matrix_processed.tsv.gz
wget https://github.com/greenelab/BioBombe/raw/f0ce9d2d91849a703b5da297a837b105c7c7b4fa/0.expression-download/data/train_gtex_expression_matrix_processed.tsv.gz
# retrieve GTEX annotations from google:
wget https://storage.googleapis.com/gtex_analysis_v7/annotations/GTEx_v7_Annotations_SubjectPhenotypesDS.txt
wget https://storage.googleapis.com/gtex_analysis_v7/annotations/GTEx_v7_Annotations_SampleAttributesDS.txt
## convert the inputDataJabba_main.rda and data.rda into csv's similar to the public data
popd
Rscript ./setup/rda2csv.R --input_dir ${RAW_DATA_ROOT} --output_dir ${RAW_DATA_ROOT}

##
# 3. Link/decompress raw data into processing directories
##
export PRETRAIN_ROOT=${MODEL_ROOT}/${PRETRAIN_PATH}
# gtex 
export GTEX_MODEL_ROOT=${PRETRAIN_ROOT}/gtex
mkdir -p ${GTEX_MODEL_ROOT}
pushd ${GTEX_MODEL_ROOT}
#gunzip -c ${RAW_DATA_ROOT}/test_gtex_expression_matrix_processed.tsv.gz  > test_gtex_expression_matrix_processed.tsv
#gunzip -c ${RAW_DATA_ROOT}/train_gtex_expression_matrix_processed.tsv.gz > train_gtex_expression_matrix_processed.tsv
#ln -s ${RAW_DATA_ROOT}/GTEx_v7_Annotations_SampleAttributesDS.txt
#ln -s ${RAW_DATA_ROOT}/GTEx_v7_Annotations_SubjectPhenotypesDS.txt
for file in test_gtex_expression_matrix_processed.tsv.gz \
    train_gtex_expression_matrix_processed.tsv.gz \
    GTEx_v7_Annotations_SampleAttributesDS.txt \
    GTEx_v7_Annotations_SubjectPhenotypesDS.txt ; 
do 
    ln -s ${RAW_DATA_ROOT}/${file}
done
popd
# tcga
export TCGA_MODEL_ROOT=${PRETRAIN_ROOT}/tcga
mkdir -p ${TCGA_MODEL_ROOT}
pushd ${TCGA_MODEL_ROOT}
#gunzip -c ${RAW_DATA_ROOT}/test_tcga_expression_matrix_processed.tsv.gz  > test_tcga_expression_matrix_processed.tsv
#gunzip -c ${RAW_DATA_ROOT}/train_tcga_expression_matrix_processed.tsv.gz > train_tcga_expression_matrix_processed.tsv
#ln -s ${RAW_DATA_ROOT}/tcga_sample_identifiers.tsv
for file in test_tcga_expression_matrix_processed.tsv.gz \
    train_tcga_expression_matrix_processed.tsv.gz \
    tcga_sample_identifiers.tsv ;
do
    ln -s ${RAW_DATA_ROOT}/${file}
done

popd
# main
## link the outputs into the 'main' analysis directory
export MAIN_MODEL_ROOT=${PRETRAIN_ROOT}/main
mkdir -p ${MAIN_MODEL_ROOT}
pushd ${MAIN_MODEL_ROOT}
for file in microarrayASDLabels_Test.csv   \
    microarrayASDLabels_Train.csv   \
    microarrayASD_Test.csv   \
    microarrayASD_Train.csv ;
do
    ln -s ${RAW_DATA_ROOT}/${file}
done
popd

##
# 4. merge.py: Convert each tsv to a processed h5, then merge under a common geneset into a single, one-hot-encoded csv
##
python ./setup/merge.py --working_dir ${PRETRAIN_ROOT} --data_dir ${DATA_ROOT} --verbose
