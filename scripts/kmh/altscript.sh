#!/bin/bash

# Input: (model) (train corpus size) (word width) (training method) (# noise
# # samples) (margin) (gamma)
export OMP_NUM_THREADS=24

ARRAY=()

# datadir="/data/taipan/karher/Corpora/Europarl/parallel"
datadir="/data/taipan/karher/oxlm/sample"
traindata="mini"
testdata="mini"
wordwidth=28

for step in 0.05
do
  for l2 in 1
  do
    for its in 1 2 3 4 5 6 7 8
    do
      for nonlinearity in "false"
      do
        for order in 2
        do
          ARRAY+=("${datadir} ${traindata} ${testdata} ${wordwidth} ${step} ${l2} ${its} ${nonlinearity} ${order}")
        done
      done
    done
  done
done

IFS=$'\n'
# declare -p ARRAY

# for i in ${ARRAY[@]}; do echo $i; done | xargs -i --max-procs 16 ./_altrunner.sh {}
for i in ${ARRAY[@]}; do ./_altrunner.sh ${i} ; done
