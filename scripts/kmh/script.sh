#!/bin/bash

# Input: (model) (train corpus size) (word width) (training method) (# noise
# # samples) (margin) (gamma)
export OMP_NUM_THREADS=24

ARRAY=()

datadir="/data/taipan/karher/Corpora/Europarl/parallel"
traindata="euro.train.10k"
testdata="euro.dev.full"
wordwidth=128

for step in 0.05 0.1 0.5
do
  for l2 in 0 0.01 0.1 1
  do
    for its in 10 25
    do
      for nonlinearity in "true" "false"
      do
        for order in 3 4
        do
          ARRAY+=("${datadir} ${traindata} ${testdata} ${wordwidth} ${step} ${l2} ${its} ${nonlinearity} ${order}")
        done
      done
    done
  done
done

IFS=$'\n'
# declare -p ARRAY

for i in ${ARRAY[@]}; do echo $i; done | xargs -i --max-procs 16 ./_runner.sh {}
