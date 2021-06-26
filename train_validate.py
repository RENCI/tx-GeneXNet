import argparse
import os
import pandas as pd
import numpy as np
import gc

import args

opt=args.getArgs()
if opt.modelToStart == "":
  g_caller = "naive"
else:
  g_caller = "transfer"
  print("[main] TRANSFER LEARNING from "+str(opt.modelToStart)+", lock ("+str(opt.finalMainUpTo)+") layers")
g_xkey="xval"
g_ykey="yval"
if opt.test:
  g_xkey="xtest"
  g_ykey="ytest"

# parameterize these?: xxx
g_evaluateTest = False

import training_functions
all_data = training_functions.loadData(opt)

# under sample our main dataset
#from imblearn.under_sampling import RandomUnderSampler
#from numpy.random import seed
# xxx should try to balance the validation data; maybe use some of the unbalanced training data?
# stopSamples=20
# idx_zeds=all_data["main"]["yval"][:stopSamples].loc[all_data["main"]["yval"]["labels"]==0].index
# xval_zeds = all_data["main"]["xval"][:stopSamples].loc[idx_zeds]
# len(xval_zeds)
# 17x0's, 23x1's
# ------------------------------------------------------------------

allValRes=[] # collect the auc results from validation
allValPreds=[] # collect the actual predictions from rounds - help us compute statistical tests later if we want
targets=[0,1]
all_data[opt.src]['ytrain']=np.asarray(all_data[opt.src]['ytrain']).astype('float32')
all_data[opt.src][g_ykey]=np.asarray(all_data[opt.src][g_ykey]).astype('float32')

data_dict = all_data[opt.src]
osize=data_dict["ytrain"].shape[1]
g_caller = "naive"
if opt.modelToStart != "":
  g_caller = "transfer"
npath=training_functions.getFPath(opt, g_caller, osize)
print('[main] Creating model: '+npath+".h5")
print('[main] Training data shape: ')
print('[main] '+opt.src+". Target size:"+str(osize))
X=data_dict["xtrain"]
y=data_dict["ytrain"]
print('[main] Training data shape: ')
print("[main] X.shape="+str(X.shape))
print("[main] y.shape="+str(y.shape))

if g_caller == "transfer":
  mpath = opt.modelToStart+".h5"
  print('[main] modelToStart - mpath='+mpath)
  print("[main] base: ")
  from tensorflow.keras.models import load_model
  base_model=load_model(mpath)
  weights_list=base_model.get_weights()
  print(base_model.summary())

model_structure=None
for i in range(opt.nRounds):
  print("[main] Round %i" % i)
  model=training_functions.getModel(opt, X, y)
  print("[main] created: ")
  print(model.summary())

  if g_caller == "transfer":
    print('[main] Setting trainable layers')
    for i in range(opt.finalMainUpTo):
      model.layers[i].set_weights(base_model.layers[i].get_weights())
      model.layers[i].trainable=False
    for i in range(opt.finalMainUpTo, len(model.layers)):
      model.layers[i].trainable=True
      print("[main] model summary: ")
      print(model.summary())
        
  print("[main] Fitting the model (patience="+str(opt.patience)+")")
  wpath, model, history, epochs = training_functions.fitModel(opt, data_dict, model, npath)
  print("[main] EVALUATION (epochs="+str(epochs)+")")
  validationResult, valPreds=training_functions.evaluate(opt, data_dict, model, wpath, targets)
  validationResult["epochs"]=epochs
  allValRes.append(validationResult)
  if model_structure == None:
    # only do this once, to print out trainable parameters at the end
    model_structure = model 
  del model
  del history
  del valPreds
  gc.collect()

print("[main] val results")
for valRes in allValRes:
  print(valRes)

training_functions.reportPerformance(opt, allValRes, g_caller, model_structure, npath)

