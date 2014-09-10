OxLM
====

Oxford Neural Language Modelling Toolkit.

### Getting started

#### Dependecies

[Cmake](http://www.cmake.org/) is needed to build the toolkit. The external dependencies are [Boost](http://www.boost.org/) and [OpenMP](http://en.wikipedia.org/wiki/OpenMP). Cmake looks for Boost and OpenMP in the locations where the libraries are installed by default using the operating system's package management tool.

To use our models with [cdec](http://www.cdec-decoder.org/), you must first download the `cdec` repository and set the `$CDEC` environment variable to point to the location where it was downloaded.

To use our models with [Moses](http://www.statmt.org/moses/), you must download the `Moses` repository first.

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

Replace the words occuring less than `min-freq` times (`min-freq` > 1) in the training data as well as the words in the development data which do not occur in the training data with the `<UNK>` symbol:

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

Unless your vocabulary is really small, you probably want to look at factored models instead.

#### Train a factored model

Partition the vocabulary using [agglomerative Brown clustering](https://github.com/percyliang/brown-cluster):

    brown-cluster/wcluster --c num-clusters \
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

#### cdec

To incorporate our neural language models as a normalized feature in the `cdec` decoder (in the beam search), simply edit the `cdec.ini` configuration file to include the following line:

    feature_function=External oxlm/lib/libcdec_ff_lbl.so --file model.bin --name LM2 --type model-type

`model-type` must be set to 1 if you are using a standard language model, 2 for factored language models and 3 for factored models with direct n-gram features.

#### Moses

Similarly, if you want to incorporate our language models in the `Moses` decoder, you first need to compile `Moses` as follows:

    ./bjam --with-lbllm=<path/to/oxlm>

You also need to specify the feature in the `Moses` configuration file under the `[feature]` section:

    FeatureType name=LM1 path=model.bin order=5

where `FeatureType` is one of `LBLLM-LM`, `LBLLM-FactoredLM` or `LBLLM-FactoredMaxentLM`. You must also specify the initial language model weight under the `[weight]` section:

    LM1= 0.5

#### Persistent caching

If you want a 2-5x speed up when tuning the translation system weights by maintaing a persistent cache of language model probabilities between consecutive iterations of the tuning algorithm, add `--persistent-cache=true` (`cdec`) or `persistent-cache=true` (`Moses`) to the decoder configuration file. Note that the persistent caching is likely to use a few GBs of disk space.
