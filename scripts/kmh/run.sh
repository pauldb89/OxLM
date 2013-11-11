#!/bin/bash

datadir=$1
trainprefix=$2
testprefix=$3
wordwidth=$4

date=$(date +%Y%m%d-%H)
expdir=../../experiments/kmh/${date}_${trainprefix}_${testprefix}

./europarl-split.sh ${datadir} ${expdir} ${trainprefix} ${testprefix}

# Run training script.
../../bin/train_cnlm \
  -s ${expdir}/data/${trainprefix}_joint_source \
  -t ${expdir}/data/${trainprefix}_joint_target \
  --l2 1 \
  --step-size 0.1 \
  --word-width ${wordwidth} \
  --no-source-eos \
  --model-out ${expdir}/${trainprefix}.model

# Train on test data with frozen weights
../../bin/train_cnlm \
  -s ${expdir}/data/${testprefix}_retrain_source \
  -t ${expdir}/data/${testprefix}_retrain_target \
  --no-source-eos \
  --model-in ${expdir}/${trainprefix}.model \
  --model-out ${expdir}/${testprefix}.retrain.model \
  --updateT false \
  --updateC false \
  --updateR false \
  --updateQ false \
  --updateF false \
  --updateFB false \
  --updateB false

# Evaluate paraphrases
../../bin/pp-logprob \
 --no-sentence-predictions \
 -l ${expdir}/data/${testprefix}_source_test_candidates \
 -s ${expdir}/data/${testprefix}_source_test_data \
 -r ${expdir}/data/${testprefix}_source_test_reference \
 -m ${expdir}/${testprefix}.retrain.model
