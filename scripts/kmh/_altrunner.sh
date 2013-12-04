#!/bin/bash

echo $1
set -- $1
datadir=$1
shift; trainprefix=$1
shift; testprefix=$1
shift; wordwidth=$1
shift; stepsize=$1
shift; l2=$1
shift; l2s=$1
shift; iterations=$1
shift; iterations2=$1
shift; nonlinearbool=$1
shift; order=$1

date=$(date +%Y%m%d-%H)
expdir=../../experiments/kmh/Z${date}_${trainprefix}_${testprefix}_${wordwidth}.s${stepsize}.l${l2}.${l2s}.i${iterations}.${iterations2}.${nonlinearbool}.${order}

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
  --source-l2 ${l2s} \
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
  --source-l2 ${l2s} \
  --step-size ${stepsize} \
  --iterations ${iterations2} \
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

for penalty in 0.1 0.2 0.3 1 2 3 4 5 6 7 8 9 10
do
  # Evaluate paraphrases
  ../../bin/pp-altprob \
    --no-sentence-predictions \
    --raw-scores \
    -l ${expdir}/data/${testprefix}_source_alttest_candidates \
    -s ${expdir}/data/${testprefix}_source_alttest_symbols \
    -t ${expdir}/data/${testprefix}_source_alttest_targets \
    -m ${expdir}/${testprefix}.retrain.model \
    --word-insertion-penalty ${penalty} \
    | tee ${expdir}/result.pen${penalty}
done
