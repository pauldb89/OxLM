#!/bin/bash

# Run training script.
../../bin/train_cnlm \
  -s euro.train.1k_source \
  -t euro.train.1k_target \
  --l2 1 \
  --step-size 0.1 \
  --word-width 128 \
