#pragma once

#include <string>

#include <boost/shared_ptr.hpp>

#include "corpus/corpus.h"
#include "lbl/config.h"
#include "lbl/utils.h"
#include "lbl/vocabulary.h"
#include "lbl/parallel_vocabulary.h"
#include "lbl/parallel_corpus.h"

// Helper functions for reading data, evaluating models, etc.

namespace oxlm {

vector<int> scatterMinibatch(const vector<int>& minibatch);

void loadClassesFromFile(
    const string& class_file, const string& training_file,
    vector<int>& classes, const boost::shared_ptr<Vocabulary>& vocab,
    VectorReal& class_bias);

void frequencyBinning(
    const string& training_file, int num_classes,
    vector<int>& classes, const boost::shared_ptr<Vocabulary>& vocab,
    VectorReal& class_bias);

int convert(
    const string& file, const boost::shared_ptr<Vocabulary>& vocab,
    bool immutable_dict, bool convert_unknowns);

boost::shared_ptr<Corpus> readTrainingCorpus(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<Vocabulary>& vocabulary,
    bool immutable_dict = true,
    bool convert_unknowns = false);

boost::shared_ptr<Corpus> readTestCorpus(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<Vocabulary>& vocabulary,
    bool immutable_dict = true,
    bool convert_unknowns = false);

int convertSource(
    const string& token,
    const boost::shared_ptr<ParallelVocabulary>& vocab);

Real perplexity(Real log_likelihood, size_t corpus_size);

} // namespace oxlm
