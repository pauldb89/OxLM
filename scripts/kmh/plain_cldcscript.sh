#!/bin/bash

# Input: (model) (train corpus size) (word width) (training method) (# noise
# # samples) (margin) (gamma)
export OMP_NUM_THREADS=24

ARRAY=()

datadir="/data/taipan/karher/Corpora/Europarl/final"
traindata="euro.train.500k.de-en"
testdata="zero"
wordwidth=64
threads=12

for step in 0.05
do
  for l2 in 1
  do
    for its in 5
    do
      for nonlinearity in "false"
      do
        for order in 2
        do
          ARRAY+=("${datadir} ${traindata} ${testdata} ${wordwidth} ${step} ${l2} ${its} ${nonlinearity} ${order} ${threads}")
        done
      done
    done
  done
done

IFS=$'\n'
# declare -p ARRAY

for i in ${ARRAY[@]}; do echo $i; done | xargs -i --max-procs 16 ./_train_cldc.sh {}
# for i in ${ARRAY[@]}; do ./_train_cldc.sh ${i} ; done
