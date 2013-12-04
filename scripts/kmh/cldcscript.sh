#!/bin/bash

# Input: (model) (train corpus size) (word width) (training method) (# noise
# # samples) (margin) (gamma)
export OMP_NUM_THREADS=24

ARRAY=()

datadir="/data/taipan/karher/Corpora/Europarl/parallel"
traindata="euro.train.1k"
testdata="euro.dev.full"
wordwidth=10

for step in 0.05
do
  for l2 in 0
  do
    for its in 1
    do
      for nonlinearity in "false"
      do
        for order in 3
        do
          ARRAY+=("${datadir} ${traindata} ${testdata} ${wordwidth} ${step} ${l2} ${its} ${nonlinearity} ${order}")
        done
      done
    done
  done
done

IFS=$'\n'
# declare -p ARRAY

for i in ${ARRAY[@]}; do echo $i; done | xargs -i --max-procs 16 ./cldcrun.sh {}
# for i in ${ARRAY[@]}; do ./cldcrun.sh ${i} ; done
