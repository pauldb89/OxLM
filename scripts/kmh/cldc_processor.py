#!/usr/bin/python
import sys, getopt
from collections import defaultdict

"""
Creates three files for the CLDC task:
  1) Target file (basically all files in the filelist concatenated
  2) Source file (naming scheme: L2D10S3 (label: 2, document: 10, sentence: 3
  3) Source with Label file: "2 L2D10S3" (later used in conversion)
"""
def usage():
  print 'cldc_processor.py [--appendix string] [--datadir string] [--postfix string] file_list dest_prefix language'

appendix = "_source"
datadir = "/data/taipan/karher/deep-embeddings/data/alex/tmp"
postfix = "train"
try:
  opts, args = getopt.getopt(sys.argv[1:], "had:", ["help","appendix=","datadir="])
except getopt.GetoptError:
  usage()
  sys.exit(2)
for opt, arg in opts:
  if opt in ("-h", "--help"):
    usage()
    sys.exit()
  elif opt in ('-a', "--appendix"):
    appendix = arg
  elif opt in ('-p', "--postfix"):
    postfix = arg
  elif opt in ('-d', "--datadir"):
    datadir = arg

filelist=open(args[0],'r')
destprefix=args[1]
language=args[2]
word_freq = defaultdict(int)

# Prepare output files:
target = open('%s_%s_target'%(destprefix,language),'w')
source = open('%s_%s_source'%(destprefix,language),'w')
labels = open('%s_%s_labels'%(destprefix,language),'w')

doc_count = 0

for actualfile in filelist:
  doc_count += 1
  sent_count = 0
  actualfile = actualfile.strip()
  file=open("%s/%s/%s"%(datadir,language,actualfile),'r')
  label_str = actualfile[-5] # xxx-L.txt
  label_int = -1
  if label_str == "M":
    label_int = 1
  if label_str == "G":
    label_int = 2
  if label_str == "E":
    label_int = 3
  if label_str == "C":
    label_int = 4
  for line in file:
    sent_count += 1
    tokens = line.split()
    for i in range(0,len(tokens)):
      tokens[i] = "%s%s"%(tokens[i],appendix)
    line_appendix = ' '.join(tokens)
    target.write("%s\n"%line_appendix)
    source.write("L%dD%dS%d_%s_%s\n"%(label_int,doc_count,sent_count,language,postfix))
    labels.write("L%dD%dS%d_%s_%s"%(label_int,doc_count,sent_count,language,postfix))
    labels.write("L%dD%dS%d_%s_%s\n"%(label_int,doc_count,sent_count,language,postfix))
