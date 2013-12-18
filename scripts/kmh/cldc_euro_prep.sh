#!/bin/bash

datadir=$1
expdir=$2
trainprefix=$3
# testprefix=$4
rcvdir=$4
rcvsize=$5

# Step 0: Make directories
mkdir -p ${expdir}/data
mkdir -p ${expdir}/cldc

#Step 1a: Combine Europarl training data and RCV for unknown word learning.
cat ${datadir}/${trainprefix}.en > ${expdir}/data/en.tmp
cat ${datadir}/${trainprefix}.de > ${expdir}/data/de.tmp
cat ${rcvdir}/train/en.${rcvsize}_en_target >> ${expdir}/data/en.tmp
cat ${rcvdir}/train/de.${rcvsize}_de_target >> ${expdir}/data/de.tmp

python ../map_unknowns.py -u _UNK_EN_ ${expdir}/data/en.tmp > ${expdir}/data/en.co.tmp
python ../map_unknowns.py -u _UNK_DE_ ${expdir}/data/de.tmp > ${expdir}/data/de.co.tmp

# Step 1b: Remove unknown words from training data (conditioned on joint data).
python ../map_unknowns.py -u _UNK_EN_ ${expdir}/data/en.co.tmp ${datadir}/${trainprefix}.en > ${expdir}/data/${trainprefix}.en.cutoff.txt
python ../map_unknowns.py -u _UNK_DE_ ${expdir}/data/de.co.tmp ${datadir}/${trainprefix}.de > ${expdir}/data/${trainprefix}.de.cutoff.txt

# Step 1c: Remove unknown words from RCV training data (conditioned on joint data).
python ../map_unknowns.py -u _UNK_EN_ ${expdir}/data/en.co.tmp ${rcvdir}/train/en.${rcvsize}_en_target > ${expdir}/data/${trainprefix}_rcv${rcvsize}_en
python ../map_unknowns.py -u _UNK_DE_ ${expdir}/data/de.co.tmp ${rcvdir}/train/de.${rcvsize}_de_target > ${expdir}/data/${trainprefix}_rcv${rcvsize}_de

# Step 1d: Remove unknown words from RCV test data (conditioned on joint data).
python ../map_unknowns.py -u _UNK_EN_ ${expdir}/data/en.co.tmp ${rcvdir}/test/en_en_target > ${expdir}/data/${trainprefix}_rcv_test_en
python ../map_unknowns.py -u _UNK_DE_ ${expdir}/data/de.co.tmp ${rcvdir}/test/de_de_target > ${expdir}/data/${trainprefix}_rcv_test_de

# Step 3: From training data, generate a joint target with a shared source, such
# that S0 -> English Sentence One and S0 -> German Sentence One.
python ../jointPreformatter.py \
  -s ${expdir}/data/${trainprefix}.en.cutoff.txt \
  -t ${expdir}/data/${trainprefix}.de.cutoff.txt \
  -o ${expdir}/data \
  --output-source-file ${trainprefix}_joint_source \
  --output-target-file ${trainprefix}_joint_target
  # --source-eos-label _EOS_EN_ \
  # --target-eos-label _EOS_DE_

# Steps 7++ (special for CLDC)

# Add training data to joint source and target
cat ${rcvdir}/train/en.${rcvsize}_en_source >> ${expdir}/data/${trainprefix}_joint_source
cat ${expdir}/data/${trainprefix}_rcv${rcvsize}_en >> ${expdir}/data/${trainprefix}_joint_target
cat ${rcvdir}/train/de.${rcvsize}_de_source >> ${expdir}/data/${trainprefix}_joint_source
cat ${expdir}/data/${trainprefix}_rcv${rcvsize}_de >> ${expdir}/data/${trainprefix}_joint_target

# Merge RCV sources and targets for frozen joint training later on
cat ${rcvdir}/train/en.${rcvsize}_en_source > ${expdir}/data/rcv_frozen_source
cat ${expdir}/data/${trainprefix}_rcv${rcvsize}_en > ${expdir}/data/rcv_frozen_target
cat ${rcvdir}/train/de.${rcvsize}_de_source >> ${expdir}/data/rcv_frozen_source
cat ${expdir}/data/${trainprefix}_rcv${rcvsize}_de >> ${expdir}/data/rcv_frozen_target

cat ${rcvdir}/test/en_en_source >> ${expdir}/data/rcv_frozen_source
cat ${expdir}/data/${trainprefix}_rcv_test_en >> ${expdir}/data/rcv_frozen_target
cat ${rcvdir}/test/de_de_source >> ${expdir}/data/rcv_frozen_source
cat ${expdir}/data/${trainprefix}_rcv_test_de >> ${expdir}/data/rcv_frozen_target
