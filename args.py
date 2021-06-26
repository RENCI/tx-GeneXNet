import sys
import argparse
import os

def _str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def printArgs(args, caller, fh=sys.stdout):
  fh.write("[printArgs] "+caller+"\n")
  for k,v in sorted(vars(args).items()): 
    fh.write("[printArgs]CONFIG: {0}={1}\n".format(k,v))


def getArgs(caller="", fileno=sys.stdout):
  parser = argparse.ArgumentParser(description='Train and validate a naive or pre-trained classifier')
  parser.add_argument('-e', '--epochs', default=200, action='store', type=int)
  parser.add_argument('-r', '--lr', default=0.0001, action='store', type=float,
                      help="Learning rate")
  parser.add_argument('-nodeFactor', '--nodeFactor', default=1.000, action='store', type=float,
                      help="Multiply number of nodes in a hidden layer by this amount; use to increase/decrease layer sizes") 

  parser.add_argument('-y', '--n_hidden', default=2, action='store', type=int,
                      help="Number of hidden layers to train") 

  # if  specified, transfer weights and lock layersbased on findalMainUpTo
  parser.add_argument('-modelToStart', '--modelToStart', default='', action='store', type=str,
                      help="Name of model to transfer from; no transfer if absent or empty string")
  parser.add_argument('-finalMainUpTo', '--finalMainUpTo', default=8, action='store', type=int,
                      help="When transferring, how many layers in the network to not train. E.g. if finalMainUpTo=8, we do not retrain the first 8 layers, which by notation of keras means the first 2 hidden layers. Hence we only train the final output layer (layer 9).")

#  parser.add_argument('-d', '--data_path', default=os.path.join('./raw'), action='store', type=str)
  parser.add_argument('-write', "--write",
                      type=_str2bool, nargs='?', const=True, default=False,
                      help="Write model out or not") # xxxI'm not sure this works; early stopping always writes the model
  parser.add_argument('-modelDir', '--modelDir', default='../model', action='store', type=str,
                      help="Directory to write model")
  parser.add_argument('-dataDir', '--dataDir', default='../setData', action='store', type=str)

  parser.add_argument('-nRounds', '--nRounds', default=50, action='store', type=int,
                      help="Number of times to repeat the training in order to compute the mean, std of the AUC on the validation data of the method")
  parser.add_argument('-stopSamples', '--stopSamples', default=0, action='store', type=int,
                      help="Set to 0 to turn off early stopping")
  parser.add_argument('-patience', '--patience', default=20, action='store', type=int)  # xxx
  parser.add_argument('-lrPatience', '--lrPatience', default=2, action='store', type=int)  # xxx
  parser.add_argument('-src', '--src', default='gtex', action='store', type=str)
  parser.add_argument('-target', '--target', default='SMTSD', action='store', type=str)
  parser.add_argument('-dryrun', '--dryrun',
                      type=_str2bool, nargs='?', const=True, default=False,
                      help="Execute commands that have no side effects")
  parser.add_argument('-test', '--test', 
                      type=_str2bool, nargs='?', const=True, default=False,
                      help="Use test data, use 'val' otherwise")

  # src=gtex, target=SMTSD
  # src=multi-tcga, target=cancer_type
  # src=main, target=labels
  parser.add_argument('-batchSize', '--batchSize', default=10, action='store', type=int)
  parser.add_argument('-smallDatasetMinEpochs', '--smallDatasetMinEpochs', default=175, action='store', type=int)

  # parameterize these?: xxx
  
  args = parser.parse_args() 
  if args.modelToStart != "":
    caller = "transfer"

  printArgs(args, caller, sys.stdout)
    
  return args
