#!/bin/bash

datadir=$1
expdir=$2
trainprefix=$3
testprefix=$4

# trainprefix="euro.train.1k"
# testprefix="euro.dev.full"

# Step 0: Make directories
mkdir -p ${expdir}/data

# Step 1: Remove singletons from training data.
python ../map_unknowns.py -u _UNK_EN_ ${datadir}/${trainprefix}.en.txt > ${expdir}/data/${trainprefix}.en.cutoff.txt
python ../map_unknowns.py -u _UNK_DE_ ${datadir}/${trainprefix}.de.txt > ${expdir}/data/${trainprefix}.de.cutoff.txt

# Step 2: Remove unknown wors from test data conditional on training data.
python ../map_unknowns.py -u _UNK_EN_ ${datadir}/${trainprefix}.en.txt ${datadir}/${testprefix}.en.txt > ${expdir}/data/${testprefix}.en.cutoff.txt
python ../map_unknowns.py -u _UNK_DE_ ${datadir}/${trainprefix}.de.txt ${datadir}/${testprefix}.de.txt > ${expdir}/data/${testprefix}.de.cutoff.txt

# Step 3: From training data, generate a joint target with a shared source, such
# that S0 -> English Sentence One and S0 -> German Sentence One.
python ../jointPreformatter.py \
  ${expdir}/data/${trainprefix}.en.cutoff.txt \
  ${expdir}/data/${trainprefix}.de.cutoff.txt \
  -o ${expdir}/data \
  -s ${trainprefix}_joint_source \
  -t ${trainprefix}_joint_target \
  --target-eos-label _EOS_

# Step 4: From test data, generate a joint target with a unique source, such
# that us0 -> English Sentence One and ut0 -> German Sentence One.
python ../uniquePreformatter.py \
  ${expdir}/data/${testprefix}.en.cutoff.txt \
  ${expdir}/data/${testprefix}.de.cutoff.txt \
  -o ${expdir}/data \
  -p ${testprefix} \
  --target-eos-label _EOS_

# Step 5: Add the test source labels to the training source, so these symbols
# have been encountered.
cat ${expdir}/data/${testprefix}_retrain_source >> ${expdir}/data/${trainprefix}_joint_source

# Step 6: Add a _FAKE_ target for each of these sources to the training data.
x=$(wc -l ${expdir}/data/${testprefix}_retrain_source | awk '{print $1}')
for (( i=0; i<x; i++ ))
do
  echo _FAKE_ >> ${expdir}/data/${trainprefix}_joint_target
done


