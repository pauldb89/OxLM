OxLM
====

Oxford Neural Language Modelling Toolkit.

### Getting started

#### Dependecies

To use our models with [cdec](http://www.cdec-decoder.org/), you must first set `$CDEC` to point to the `cdec` repository.

#### Installation

Run the following to compile the code for the first time:

    cd oxlm
    mkdir build
    cd build
    cmake ../src
    make

Run unit tests:

    cd build
    make all_tests

### Prepare the training data

Replace the words occuring less than `min-freq` times in the training data as well as the words in the development data which do not occur in the training data with the `<UNK>` symbol:

    sh oxlm/scripts/countcutoff.sh training.en min-freq
    python oxlm/scripts/preprocess-corpus.py -i training.en,dev.en -o training.unk.en,dev.unk.en -v vocab

### Training

#### Train a standard model

Create a `oxlm.ini` file with the following contents:

    iterations=20
    minibatch-size=10000
    lambda-lbl=2
    word-width=100
    step-size=0.06
    order=5
    randomise=true
    diagonal-contexts=true
    sigmoid=true

    input=training.unk.en
    test-set=dev.unk.en

Run:

    oxlm/bin/train_sgd -c oxlm.ini --threads=8 --model-out=model.bin

Set the `--noise-samples` argument, if you want to train the models using noise contrastive estimation instead of minibatch stochastic gradient descent.

#### Train a factored model

Partition the vocabulary using [agglomerative Brown clustering](https://github.com/percyliang/brown-cluster):

    brown-cluster/wcluster -c num-clusters \
                           --text training.unk.en \
                           --output_dir=clusters

Set `num-clusters` to the square root of the size of the vocabulary. To train the model, run:

    oxlm/train_factored_sgd -c oxlm.ini \
                            --threads=8 \
                            --model-out=model.bin \
                            --class-file=clusters/path

#### Train a factored model with direct n-gram features

Append the following to the `oxlm.ini` configuration file:

    sparse-features=true
    feature-context-size=5
    min-ngram-freq=2
    filter-contexts=true

Run the following command to train the model:

    oxlm/train_maxent_sgd -c oxlm.ini \
                          --threads=8 \
                          --model-out=model.bin \
                          --class-file=clusters/path

This will use a one-to-one mapping from features to weights. If you want to use a lower dimensional feature space for the weights (i.e. collision stores), use the `--hash-space` parameter. Generally, setting the hash-space to 1/2 or 1/4 of the total number of features results in a negligible loss in perplexity. Collision store reduce the memory requirements significantly.

### Decoding

To incorporate our neural language models as a normalized feature in the `cdec` decoder (in the beam search), simply edit the `cdec.ini` configuration file to include the following line:

    feature_function=External oxlm/lib/libcdec_ff_lbl.so --file model.bin --name LM2 --type model-type

`model-type` must be set to 1 if you are using a standard language model, 2 for factored language models and 3 for factored models with direct n-gram features.
