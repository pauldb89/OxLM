#!/bin/bash

echo $1
set -- $1
datadir=$1
shift; trainprefix=$1
shift; testprefix=$1
shift; wordwidth=$1
shift; stepsize=$1
shift; l2=$1
shift; iterations=$1
shift; nonlinearbool=$1
shift; order=$1

date=$(date +%Y%m%d-%H)
expdir=../../experiments/kmh/${date}_${trainprefix}_${testprefix}

nonlinear=""
if [ "$nonlinearbool" == "true" ]
then
  nonlinear="--non-linear"
fi

./europarl-split.sh ${datadir} ${expdir} ${trainprefix} ${testprefix}

# Run training script.
../../bin/train_cnlm \
  -s ${expdir}/data/${trainprefix}_joint_source \
  -t ${expdir}/data/${trainprefix}_joint_target \
  --no-source-eos \
  --l2 ${l2} \
  --step-size ${stepsize} \
  --iterations ${iterations} \
  --word-width ${wordwidth} \
  --order ${order} \
  ${nonlinear} \
  --model-out ${expdir}/${trainprefix}.model

# Train on test data with frozen weights
../../bin/train_cnlm \
  -s ${expdir}/data/${testprefix}_retrain_source \
  -t ${expdir}/data/${testprefix}_retrain_target \
  --model-in ${expdir}/${trainprefix}.model \
  --model-out ${expdir}/${testprefix}.retrain.model \
  --no-source-eos \
  --l2 ${l2} \
  --step-size ${stepsize} \
  --iterations ${iterations} \
  --word-width ${wordwidth} \
  --order ${order} \
  ${nonlinear} \
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
 -m ${expdir}/${testprefix}.retrain.model \
 > ${expdir}/result
