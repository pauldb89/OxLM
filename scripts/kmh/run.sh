#!/bin/bash

# Run training script.
../../bin/train_cnlm \
  -s euro.train.1k_source \
  -t euro.train.1k_target \
  --l2 1 \
  --step-size 0.1 \
  --word-width 128 \
  --model-out euro.train.1k.model

# Train on test data with frozen weights
../../bin/train_cnlm \
  -s euro.dev.full_testtrain_source \
  -t euro.dev.full_testtrain_target \
  --model-in euro.train.1k.model \
  --model-out euro.train.1k_euro.dev.full.testtrained.model \
  --expand-source-dict true \
  --updateT false \
  --updateC false \
  --updateR false \
  --updateQ false \
  --updateF false \
  --updateFB false \
  --updateB false

# Evaluate paraphrases
../../bin/pp-logprob \
 -l euro.dev.full_target_candidates \
 -s euro.dev.full_target_data \
 -r euro.dev.full_target_reference \
 -m euro.train.1k_euro.dev.full.testtrained.model
