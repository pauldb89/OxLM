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
shift; threads=$1

rcvdir="/data/taipan/karher/deep-embeddings/data/alex/cnlm"
rcvsize="1000"
avgperceptrondir="/data/taipan/karher/deep-embeddings/tools/avgperceptron"

date=$(date +%Y%m%d-%H)
# date="20131118-17"
expdir=${PWD}/../../experiments/kmh/${date}_CLDC_${trainprefix}_${rcvsize}

nonlinear=""
if [ "$nonlinearbool" == "true" ]
then
  nonlinear="--non-linear"
fi

./cldc_euro_prep.sh ${datadir} ${expdir} ${trainprefix} ${rcvdir} ${rcvsize}

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
  --threads ${threads} \
  ${nonlinear} \
  --model-out ${expdir}/${trainprefix}.model

# Train on RCV data with frozen weights
../../bin/train_cnlm \
  -s ${expdir}/data/rcv_frozen_source \
  -t ${expdir}/data/rcv_frozen_target \
  --model-in ${expdir}/${trainprefix}.model \
  --model-out ${expdir}/retrain.model \
  --no-source-eos \
  --replace-source-dict \
  --l2 ${l2} \
  --step-size ${stepsize} \
  --iterations ${iterations} \
  --word-width ${wordwidth} \
  --order ${order} \
  --threads ${threads} \
  ${nonlinear} \
  --updateT false \
  --updateC false \
  --updateR false \
  --updateQ false \
  --updateF false \
  --updateFB false \
  --updateB false

# Extract vectors using the source and target label files.
for lang in "de" "en"
do
  echo ../../bin/print_cldc \
    -i ${rcvdir}/train/${lang}.${rcvsize}_${lang}_labels \
    -m ${expdir}/retrain.model \
    -o ${expdir}/cldc/${lang}_train
  ../../bin/print_cldc \
    -i ${rcvdir}/train/${lang}.${rcvsize}_${lang}_labels \
    -m ${expdir}/retrain.model \
    -o ${expdir}/cldc/${lang}_train
  ../../bin/print_cldc \
    -i ${rcvdir}/test/${lang}_${lang}_labels \
    -m ${expdir}/retrain.model \
    -o ${expdir}/cldc/${lang}_test
  python cldc_merger.py ${expdir}/cldc/${lang}_train ${expdir}/cldc/${lang}_train_doc
  python cldc_merger.py ${expdir}/cldc/${lang}_test ${expdir}/cldc/${lang}_test_doc
  sed -i 's/nan/0.00/g' ${expdir}/cldc/${lang}_train_doc
  sed -i 's/nan/0.00/g' ${expdir}/cldc/${lang}_test_doc
done

# Train stuff with Java.

echo "Phase 2: Training Models"
cd ${avgperceptrondir}
for lang in "de" "en"
do
  filelist="${expdir}/cldc/${lang}_train_doc"
  xs="java ApLearn \
    --train-set ${filelist} \
    --model-name ${expdir}/cldc/${lang}.mod"
  eval $xs
done

echo "Phase 3: Evaluating Models" | tee ${expdir}/cldc.log
printf "Model\tSelf   Own    Foreign\n" | tee -a ${expdir}/cldc.log
for lang in "de" "en"
do
  antilang="de"
  if [ "$lang" == "$antilang" ]
  then
    antilang="en"
  fi
  filelist="${expdir}/cldc/${lang}_train_doc"
  printf "${lang}\t" | tee -a ${expdir}/cldc.log
  for testdata in "${filelist}" "${expdir}/cldc/${lang}_test_doc" "${expdir}/cldc/${antilang}_test_doc"
  do
    exactaccuracy=`java ApClassify \
      --test-set ${testdata} \
      --model-name ${expdir}/cldc/${lang}.mod \
      | tail -n 1 | awk '{print $2}'`
    printf "%0.3f  " ${exactaccuracy} | tee -a ${expdir}/cldc.log
  done
  echo "" | tee -a ${expdir}/cldc.log
done
