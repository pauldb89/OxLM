#pragma once

#include <string>

#include <boost/shared_ptr.hpp>

#include "corpus/corpus.h"
#include "lbl/config.h"
#include "lbl/utils.h"

// Helper functions for reading data, evaluating models, etc.

namespace oxlm {

vector<int> scatterMinibatch(int start, int end, const vector<int>& indices);

void loadClassesFromFile(
    const string& class_file, const string& training_file,
    vector<int>& classes, Dict& dict, VectorReal& class_bias);

void frequencyBinning(
    const string& training_file, int num_classes,
    vector<int>& classes, Dict& dict, VectorReal& class_bias);

int convert(
    const string& file, Dict& dict,
    bool immutable_dict, bool convert_unknowns);

boost::shared_ptr<Corpus> readCorpus(
    const string& file, Dict& dict,
    bool immutable_dict = true, bool convert_unknowns = false);

Real perplexity(Real log_likelihood, size_t corpus_size);

} // namespace oxlm
