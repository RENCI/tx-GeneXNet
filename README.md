# tx-GeneXNet
Code for transfer-learning gene expression data.

# How to run
### Step 0: Set up your environment
Set up environmental variables to make it easier to run the scripts. File `sample_env` is provided for convenience.
```
cp sample_env env
# edit env to reflect your environment
source ./env
```
We suggest configuring output data to a level outside the repo so they aren't accidentally committed. 

### Step 1: Execute the data munging pipeline
Download the pretraining source files, and setup directories and data.
```
time ./setup.sh > ../setup.out
```

If using the sample_env values, the final, prepared dataset will bin in `../setData`. The following datasets are prepared:
##### Pre-training data and targets
```
gtex-SMTSD-xtrain.csv 
gtex-SMTSD-ytrain.csv        # target/labels
```
##### Training data and targets (with ASD being the 'main' model)
```
main-labels-xtrain.csv
main-labels-ytrain.csv       # target/labels
```
##### Validation data and targets
The following data were used for selection of parameters, including learning rate, number of layers, nodes, epochs.
```
main-labels-xval.csv
main-labels-yval.csv       # target/labels
```
##### Test data and targets
The following hold-out data will be used for final performance report:
```
main-labels-xtest.csv
main-labels-ytest.csv      # target/labels
```
### Step 2: Execute the training pipeline

Sample executions can be done by the commands below.

To run the naive model (without pretraining):
```
#naive
time python train_validate.py -src main -target labels -e 200 -r 0.0001 -y 1 -write False -modelDir ../model -dataDir ../setData -stopSamples 0 \
     -nRounds 40 -test
```

To run the pretraining procedure:
```
# transfer
time python train_validate.py -src main -target labels -e 200 -r 0.0001 -y 1 -write False -modelDir ../model -dataDir ../setData  -stopSamples 0 \
     -modelToStart ../model/gtex-naive-eps:300-lr:10-nhidden:1-outputSize:31 -finalMainUpTo 4 \
     -nRounds 40  -test > ../out/test-pretrained
```

Run with `-h` option for `usage` report, including description of options.
