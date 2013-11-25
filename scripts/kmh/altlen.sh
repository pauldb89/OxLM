#!/bin/bash

# Input: (model) (train corpus size) (word width) (training method) (# noise
# # samples) (margin) (gamma)
export OMP_NUM_THREADS=24

ARRAY=()

OLDDIR=${PWD}
MYDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $MYDIR

if [ $1 == "clg" ]
then
  datadir="/data/taipan/karher/Corpora/Europarl/parallel"
else
  datadir="${PWD}/../../sample"
fi
echo $datadir
traindata="length"
testdata="length"
wordwidth=28

for step in 0.05 0.1
do
  for l2 in 0.1 0.5 1
  do
    for its in 10
    do
      for nonlinearity in "false"
      do
        for order in 2
        do
          for its2 in 10
          do
            for l2s in 0 0.01 0.1 1
            do
              ARRAY+=("${datadir} ${traindata} ${testdata} ${wordwidth} ${step} ${l2} ${l2s} ${its} ${its2} ${nonlinearity} ${order}")
            done
          done
        done
      done
    done
  done
done

IFS=$'\n'
# declare -p ARRAY

# for i in ${ARRAY[@]}; do echo $i; done | xargs -i --max-procs 16 ./_altrunner.sh {}
for i in ${ARRAY[@]}; do ./_altrunner.sh ${i} ; done

cd $OLDDIR
