# usage: Rscript rda2csv.R -i <input_dir> -o <output_dir> [--dry_run]
# 
# e.g.
#    Rscript prepMain.R --input_dir ${RAW_DATA_ROOT} --output_dir ${MAIN_MODEL_ROOT}
#
# Purpose:
#   Convert .rda files to tsv's similar in structure to BioBombe project
# Inputs:
#   inputDataJabba_main.rda, data.rda
# Outputs:
#   microarrayASDLabels_Test.csv   
#   microarrayASDLabels_Train.csv   
#   microarrayASD_Test.csv   
#   microarrayASD_Train.csv ;
#

# install.packages("optparse")
library(optparse)
option_list = list(
    make_option(c("-o", "--output_dir"), type="character", default=NULL,
              help="container (transient) output directory, omit for stdout  [default= %default]", metavar="character"),
    make_option(c("-i", "--input_dir"), type="character", default=NULL, 
              help="user-defined participant groupname created on Immunespace [default= %default]", metavar="character"),
    make_option(c("-d", "--dry_run"), action="store_true", default=FALSE,
              help="Don't execute anything just print commands [default %default]", metavar="character")
); 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);
if (is.null(opt$output_dir) || is.null(opt$input_dir)){
  print_help(opt_parser)
  stop("Must supply input and output directories", call.=FALSE)
}
# ---- Variables ---
inputDir <- opt$input_dir
outputDir <- opt$output_dir
debug <- opt$dry_run
# ---- Initialization ---
if(debug == TRUE){
  cat("== DRY RUN ==\n")
}

#array data and spit it out in csv format (for ml)
library(Biobase)

##
# convert inputDataJabba_main.rda:
##
infile=paste0(inputDir, "/", "inputDataJabba_main.rda")
print(paste0("Reading:", infile))
load(infile)

labelName="microarrayASDLabels_Train.csv"
dataName="microarrayASD_Train.csv"
##extract csv from expressionset
data=assayData(inputExpData)$exprs
data=t(data)
data=cbind(data, labels)


outputFile=paste0(outputDir, "/", dataName)
print(paste0("Writing: ", outputFile))
if(debug == TRUE){
  cat("+ write.csv(data, outputFile, row.names=F)\n")
} else {
  write.csv(data, outputFile, row.names=F)
}

outputFile=paste0(outputDir, "/", labelName)
print(paste0("Writing: ", outputFile))
if(debug == TRUE){
  cat("+   write.csv(labels, outputFile, row.names=F) \n")
} else {
  write.csv(labels, outputFile, row.names=F)
}

##
# convert data.rda:
##
# write out test files
infile=paste0(inputDir, "/", "data.rda")
print(paste0("Reading:", infile))
load(infile)
dataName="microarrayASD_Test.csv"
labelName="microarrayASDLabels_Test.csv"
colsOfInterest="geneSymbol"

# extract data, labels from rda
labels=data$testLabels
data=assayData(data$testInputData)$exprs
data=t(data)
data=cbind(data, labels)

outputFile=paste0(outputDir, "/", labelName)
print(paste0("Writing: ", outputFile))
if(debug == TRUE){
  cat("+     write.csv(labels, outputFile, row.names=F) \n")
} else {
  write.csv(labels, outputFile, row.names=F)
}

outputFile=paste0(outputDir, "/", dataName)
print(paste0("Writing: ", outputFile))
if(debug == TRUE){
  cat("+       write.csv(data, outputFile, row.names=F) \n")
} else {
  write.csv(data, outputFile, row.names=F)
}

