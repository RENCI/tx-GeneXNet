import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

from args import printArgs

def loadData(opt):
  print("[loadData] Load the datasets...")
  data={}
  data[opt.src]={}

  xkey="xval"
  ykey="yval"
  if opt.test:
    xkey="xtest"
    ykey="ytest"
  subkeys=['xtrain', 'ytrain', xkey, ykey] # xxx
  # 'ASD' has both test and val datasets
  # but tcga, gtex have only test, so use it for validation
  for subKey in subkeys:
    fn=os.path.join(opt.dataDir, opt.src+"-"+opt.target+"-"+subKey+".csv")
    if not os.path.exists(fn):
      print("[main] ERROR: file not found - ("+fn+")")
    else:
      print("[loadData] ("+opt.src+":"+subKey+"),reading:"+fn)
      data[opt.src][subKey]=pd.read_csv(fn)
  return data

def getFPath(opt, base, osize):
  return opt.modelDir+"/"+opt.src+"-"+base+"-eps:{:d}-lr:{:.0f}-nhidden:{:d}-outputSize:{:d}".format(opt.epochs, opt.lr*10000, opt.n_hidden, osize)

# ------------------------------------------------------------------
# model configuration is largely the same as what Kimberly originally suggested
def getModel(opt, X, y): #ideally use y for outputs size later,...
  model = Sequential()

  node_factor=opt.nodeFactor
  n_hidden=opt.n_hidden
  learning_rate=opt.lr
  output_size=y.shape[1]
  feature_size=X.shape[1]
  observations_size=X.shape[0]

  isMultilabel = True
  output_activation = 'softmax'
  loss = 'categorical_crossentropy'
  if output_size==1:
    isMultilabel = False 
    output_activation='sigmoid'
    loss = 'binary_crossentropy' 

  i=1
  m=round(X.shape[1]*((n_hidden-i+1)/(n_hidden+1)) * node_factor) 
  layer_name='input_1'
  print("[getModel] nodes in layer "+layer_name+": "+str(m))
  model.add(Dense(m, input_dim=X.shape[1], name=layer_name))
  model.add(BatchNormalization(name='1_decoder_batch_norm'))
  model.add(Activation('relu', name="1_activation"))
  print("[getModel] relu for 1_activation")
  model.add(Dropout(0.50, name='1_hidden_dropout'))
  for i in range(2,(n_hidden+1)):
    m=round(X.shape[1]*((n_hidden-i+1)/(n_hidden+1))* node_factor) 
    layer_name=str(i)+'_hidden'
    print("[getModel] nodes in layer "+layer_name+": "+str(m))
    model.add(Dense(m, name=layer_name))
    model.add(BatchNormalization(name=str(i)+'_decoder_batch_norm'))
    model.add(Activation('relu', name=str(i)+"_activation"))
    print("[getModel] relu for activation for "+str(i)+"_activation")
    model.add(Dropout(0.50, name=str(i)+'_hidden_dropout'))

  print("[getModel] Compiling with output size= "+str(output_size)+", activation="+output_activation+", loss="+loss)
  model.add(Dense(output_size, activation=output_activation,name = "output"))
  model.compile(loss=loss, optimizer=optimizers.Adam(learning_rate),
              metrics=['accuracy', 'mse',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.AUC(name='auc',
                                            multi_label=isMultilabel),
                       tf.keras.metrics.SpecificityAtSensitivity(0.5, name='SpecificityAtSensitivity'),
                       tf.keras.metrics.SensitivityAtSpecificity(0.5, name='SensitivityAtSpecificity'),
                       tf.keras.metrics.FalsePositives(name='fp'),
                       tf.keras.metrics.FalseNegatives(name='fn'),
                       tf.keras.metrics.TruePositives(name='tp'),
                       tf.keras.metrics.TrueNegatives(name='tn')
                   ])
  return model



def fitModel(opt, data_dict, model, npath):

  X=data_dict["xtrain"]
  y=data_dict["ytrain"]

  weights_fpath=npath+".h5" # xxx ask Hong why you would do this/where you would read in weights-best
  epochs = opt.epochs
  if opt.stopSamples > 0:
    print("[fitModel] earlyStopping, patience="+str(opt.patience)+", patience validation split="+str(opt.stopSamples))

    xkey="xval"
    ykey="yval"
    if opt.test:
      xkey="xtest"
      ykey="ytest"
    X_val=data_dict[(xkey)][:opt.stopSamples]
    y_val=data_dict[(ykey)][:opt.stopSamples]

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    #stopping_patience=patience #5,  10=number of epochs to stop after if no improvement
    #lrPatience=lrPatience       # 2=number of epochs to reduce lr after if no improvement; s/b smaller than opt.patience


    checkpoint = ModelCheckpoint(weights_fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_freq='epoch')
    earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=opt.patience) # minimize loss; 
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=opt.lrPatience) # xxx prolly not necessary for overtraining, where lr needs to be small anyway because converges quickly, but if so start learning at 0.01
    #    history = model.fit(X, y, epochs=opt.epochs, callbacks=[earlystop, reduce],
    #                        validation_data=(X_val, y_val))
    
    initial_epoch = 0
    multi_label = y.shape[1] > 1
    if not multi_label:
      num_positives = y.sum()
      num_sequences = len(y)
      num_negatives = num_sequences - num_positives
      # train model 150 epochs and save if not multi
      history = model.fit(X, y, batch_size=opt.batchSize, epochs=opt.smallDatasetMinEpochs)
      initial_epoch = opt.smallDatasetMinEpochs

    # load saved model and continue
    if not opt.dryrun:
      history = model.fit(X, y, batch_size=opt.batchSize, epochs=opt.epochs, initial_epoch= initial_epoch, validation_data=(X_val, y_val),
                          class_weight={
                            True: num_sequences / num_positives,
                            False: num_sequences / num_negatives
                          } if not multi_label else None,
                          callbacks=[earlystop, reduce, checkpoint],
                          verbose=True
      )
    else:
      print("+ history = model.fit(X, y, batch_size="+str(opt.batchSize)+", epochs="+str(opt.epochs)+", initial_epoch= "+str(initial_epoch)+
            ", validation_data=(X_val, y_val), class_weight={                             True: num_sequences / num_positives,                            False: num_sequences / num_negatives                          } if not multi_label else None,                          callbacks=[earlystop, reduce, checkpoint],                          verbose=True)")# xxx finish this
                        
    epochs = earlystop.stopped_epoch

  else:
    if not opt.dryrun:
      history = model.fit(X, y, epochs=opt.epochs, batch_size=opt.batchSize)
    else:
      print("+ model.fit(X, y, epochs=opt.epochs, batch_size=opt.batchSize)") # xxx
    if opt.write:
      if not opt.dryrun:
        model.save(weights_fpath)
        if not os.path.exists(weights_fpath):
          print("[transfer] ERROR couldn't write %s\n" % weights_fpath)
        else:
          print("[transfer] *** wrote model to %s\n" % weights_fpath)
      else:
        print("+ model.save("+weights_fpath+")")
  if not opt.dryrun:
    return weights_fpath, model, history, epochs
  else:
    return weights_fpath, model, None, epochs

# ------------------------------------------------------------------
# in evaluation, we set up the proper evaluation task, evaluate it and collect prediction for other analysis reports
def evaluate(opt, data_obj, model, filepath, targets):
  learning_rate=opt.lr
  n_hidden=opt.n_hidden

  X=data_obj["xtrain"]
  y=data_obj["ytrain"]

  xkey="xval"
  ykey="yval"
  stopSamples = opt.stopSamples
  if opt.test:
    xkey="xtest"
    ykey="ytest"
    print("[evaluate]-------- using TEST data, disablign early stopping -------")
    stopSamples = 0
  else:    
    print("[evaluate]-------- using validation data -------")
  # stopSamples are used for 'early stopping', so only evaluate the validation data AFTER stopSamples
  # default is '0' stopSamples, evaluate on all the validation data
  Xtest=data_obj[xkey][stopSamples:]
  print(Xtest.shape)
  ytest=data_obj[ykey][stopSamples:]
  print(ytest.shape)
  ytest=np.array(ytest)
  # --------------
  # collect metrics in dictionary form (metrDict)
  result = model.evaluate(Xtest, ytest)  # xxx set batch_size here - 10?
  metrDict={}
  tp, fp, tn, fn = -1, -1, -1, -1
  for i in range(len(model.metrics_names)):
    metric_name=model.metrics_names[i]
    metric=result[i]
    metrDict[metric_name]=metric
    if metric_name=='fp':
      fp=metric
    if metric_name=='tp':
      tp=metric
    if metric_name=='fn':
      fn=metric
    if metric_name=='tn':
      tn=metric
  sens=float(tp)/float(tp+fn)
  spec=float(tn)/float(tn+fp)
  metrDict['sens']=sens
  metrDict['spec']=spec
  # xxx this is maybe redundant?
  #  for k in list(metrDict.keys()):
  #    print('[evaluate] \t %s: %.2f' % (k, metrDict[k]))
  # --------------

  from sklearn import metrics
  preds = model.predict(Xtest)#[f for l in model.predict(Xtest)]

  if opt.src=="main":
    tau=.5
    preds[preds >= tau]=targets[1]
    preds[preds < tau]=targets[0]

    preds=[int(i[0]) for i in preds]
    ytest=[int(i[0]) for i in ytest]

  else:
    preds=[np.argmax(i) for i in preds]
    ytest=[np.argmax(i) for i in ytest]

  matrix = metrics.confusion_matrix(ytest, preds)
  report = metrics.classification_report(ytest, preds, output_dict=True) # add output_dict for csv file

  if opt.write:
    df = pd.DataFrame(report)
    df.transpose()

    if not opt.dryrun:
      df.to_csv(filepath+".csv")
    else:
      print("+ df.to_csv("+filepath+"+\".csv\")")
    #model.save(filepath+".h5")
    #print("*** wrote files to %s.\n" % filepath)
    print("[evaluate] *** wrote metrics to %s.\n" % (filepath + ".csv") )

  return metrDict, preds

# requires python 3.4+
from contextlib import redirect_stdout
def reportPerformance(opt, allValRes, caller, model=None, npath=""):
  print("[reportPerformance] Validation results")
  import datetime
  x = datetime.datetime.now()
  ts = x.strftime("%Y%m%d.%H%M%S")
  stats_fn = os.path.join(opt.modelDir, caller+"-"+opt.src+"-"+opt.target+"-"+ts+".csv")
  if not opt.dryrun:
    fh = open(stats_fn, "w")
    if npath != "":
      fh.write("[reportPerformance] created model: "+npath+"\n")
      printArgs(opt, caller, fh)
    if model != None:
      print("[reportPerformance] summary of last trained model: ")
      print(model.summary())
      with redirect_stdout(fh):
        model.summary()
    for valRes in allValRes:
      print(valRes)
      fh.write(str(valRes["epochs"])+","+str(valRes["auc"])+"\n")
    print("[reportPerformance] Computing summary statistics")
    _summarize(opt, fh, 'auc', allValRes)
    _summarize(opt, fh, 'epochs', allValRes)
    fh.close()
    print("[reportPerformance] wrote file: "+stats_fn)
  else:
    print("+save report performance ("+stats_fn+")")

def _summarize(opt, fh, metric, allValRes):
  aggregated=[]
  for valRes in allValRes:
    aggregated.append(valRes[metric])
  import scipy.stats
  confidence = 0.95
  a = 1.0 * np.array(aggregated)
  mu, se = np.mean(a), scipy.stats.sem(a)
  h = se * scipy.stats.t.ppf((1 + confidence) / 2., len(a)-1)
  print("[_summarize] mu:"+str(mu)+" ± "+str(h))
  print("[_summarize] SE: "+str(se))
  if not opt.dryrun:
    fh.write(str(mu)+" ± "+str(h)+"\n")
    fh.write("SE: "+str(se)+"\n")
  else:
    print("+ write SE, mu to file")
