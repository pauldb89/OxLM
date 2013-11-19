#!/bin/bash

datadir=$1
expdir=$2
trainprefix=$3
# testprefix=$4
rcvdir=$4
rcvsize=$5

# trainprefix="euro.train.1k"
# testprefix="euro.dev.full"

# Step 0: Make directories
mkdir -p ${expdir}/data
mkdir -p ${expdir}/cldc

#Step 1a: Combine Europarl training data and RCV for unknown word learning.
cat ${datadir}/${trainprefix}.en.txt > ${expdir}/data/en.tmp
cat ${datadir}/${trainprefix}.de.txt > ${expdir}/data/de.tmp
cat ${rcvdir}/train/en.${rcvsize}_en_target >> ${expdir}/data/en.tmp
cat ${rcvdir}/train/de.${rcvsize}_de_target >> ${expdir}/data/de.tmp

python ../map_unknowns.py -u _UNK_EN_ ${expdir}/data/en.tmp > ${expdir}/data/en.co.tmp
python ../map_unknowns.py -u _UNK_DE_ ${expdir}/data/de.tmp > ${expdir}/data/de.co.tmp

# Step 1b: Remove unknown words from training data (conditioned on joint data).
python ../map_unknowns.py -u _UNK_EN_ ${expdir}/data/en.co.tmp ${datadir}/${trainprefix}.en.txt > ${expdir}/data/${trainprefix}.en.cutoff.txt
python ../map_unknowns.py -u _UNK_DE_ ${expdir}/data/de.co.tmp ${datadir}/${trainprefix}.de.txt > ${expdir}/data/${trainprefix}.de.cutoff.txt

# Step 2: Remove unknown words from test data (conditioned on joint data).
# python ../map_unknowns.py -u _UNK_EN_ ${expdir}/data/en.co.tmp ${datadir}/${testprefix}.en.txt > ${expdir}/data/${testprefix}.en.cutoff.txt
# python ../map_unknowns.py -u _UNK_DE_ ${expdir}/data/de.co.tmp ${datadir}/${testprefix}.de.txt > ${expdir}/data/${testprefix}.de.cutoff.txt

# Step 1c: Remove unknown words from RCV training data (conditioned on joint data).
python ../map_unknowns.py -u _UNK_EN_ ${expdir}/data/en.co.tmp ${rcvdir}/train/en.${rcvsize}_en_target > ${expdir}/data/${trainprefix}_rcv${rcvsize}_en
python ../map_unknowns.py -u _UNK_DE_ ${expdir}/data/de.co.tmp ${rcvdir}/train/de.${rcvsize}_de_target > ${expdir}/data/${trainprefix}_rcv${rcvsize}_de

# Step 1d: Remove unknown words from RCV test data (conditioned on joint data).
python ../map_unknowns.py -u _UNK_EN_ ${expdir}/data/en.co.tmp ${rcvdir}/test/en_en_target > ${expdir}/data/${trainprefix}_rcv_test_en
python ../map_unknowns.py -u _UNK_DE_ ${expdir}/data/de.co.tmp ${rcvdir}/test/de_de_target > ${expdir}/data/${trainprefix}_rcv_test_de

# Step 3: From training data, generate a joint target with a shared source, such
# that S0 -> English Sentence One and S0 -> German Sentence One.
python ../jointPreformatter.py \
  ${expdir}/data/${trainprefix}.en.cutoff.txt \
  ${expdir}/data/${trainprefix}.de.cutoff.txt \
  -o ${expdir}/data \
  -s ${trainprefix}_joint_source \
  -t ${trainprefix}_joint_target
  # --source-eos-label _EOS_EN_ \
  # --target-eos-label _EOS_DE_

# Step 4: From test data, generate a joint target with a unique source, such
# that us0 -> English Sentence One and ut0 -> German Sentence One.
# python ../uniquePreformatter.py \
  # ${expdir}/data/${testprefix}.en.cutoff.txt \
  # ${expdir}/data/${testprefix}.de.cutoff.txt \
  # -o ${expdir}/data \
  # -p ${testprefix}
  # # --source-eos-label _EOS_EN_ \
  # # --target-eos-label _EOS_DE_

# Step 5: Add the test source labels to the training source, so these symbols
# have been encountered.
# cat ${expdir}/data/${testprefix}_retrain_source >> ${expdir}/data/${trainprefix}_joint_source

# Step 6: Add a _FAKE_ target for each of these sources to the training data.
# x=$(wc -l ${expdir}/data/${testprefix}_retrain_source | awk '{print $1}')
# for (( i=0; i<x; i++ ))
# do
  # echo _FAKE_ >> ${expdir}/data/${trainprefix}_joint_target
# done

# Steps 7++ (special for CLDC)

# Add training data to joint source and target
cat ${rcvdir}/train/en.${rcvsize}_en_source >> ${expdir}/data/${trainprefix}_joint_source
cat ${expdir}/data/${trainprefix}_rcv${rcvsize}_en >> ${expdir}/data/${trainprefix}_joint_target
cat ${rcvdir}/train/de.${rcvsize}_de_source >> ${expdir}/data/${trainprefix}_joint_source
cat ${expdir}/data/${trainprefix}_rcv${rcvsize}_de >> ${expdir}/data/${trainprefix}_joint_target

# Add test data to joint source
cat ${rcvdir}/test/en_en_source >> ${expdir}/data/${trainprefix}_joint_source
cat ${rcvdir}/test/de_de_source >> ${expdir}/data/${trainprefix}_joint_source

# Add fakes to joint target
x=$(wc -l ${rcvdir}/test/en_en_source | awk '{print $1}')
for (( i=0; i<x; i++ ))
do
  echo _FAKE_ >> ${expdir}/data/${trainprefix}_joint_target
done

x=$(wc -l ${rcvdir}/test/de_de_source | awk '{print $1}')
for (( i=0; i<x; i++ ))
do
  echo _FAKE_ >> ${expdir}/data/${trainprefix}_joint_target
done

# Merge RCV sources and targets for frozen joint training later on
cat ${rcvdir}/train/en.${rcvsize}_en_source > ${expdir}/data/rcv_frozen_source
cat ${expdir}/data/${trainprefix}_rcv${rcvsize}_en > ${expdir}/data/rcv_frozen_target
cat ${rcvdir}/train/de.${rcvsize}_de_source >> ${expdir}/data/rcv_frozen_source
cat ${expdir}/data/${trainprefix}_rcv${rcvsize}_de >> ${expdir}/data/rcv_frozen_target

cat ${rcvdir}/test/en_en_source >> ${expdir}/data/rcv_frozen_source
cat ${expdir}/data/${trainprefix}_rcv_test_en >> ${expdir}/data/rcv_frozen_target
cat ${rcvdir}/test/de_de_source >> ${expdir}/data/rcv_frozen_source
cat ${expdir}/data/${trainprefix}_rcv_test_de >> ${expdir}/data/rcv_frozen_target
