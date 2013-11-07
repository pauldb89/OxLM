#!/bin/bash

EURODIR=$1
echo $PWD
for term in "euro.train.1k" "euro.dev.full"
do
  python ../jointPreformatter.py \
    ${EURODIR}/${term}.en.txt \
    ${EURODIR}/${term}.de.txt \
    -o ${PWD} \
    -s ${term}_source \
    -t ${term}_target
done
