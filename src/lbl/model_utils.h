#pragma once

#include <string>

#include <boost/shared_ptr.hpp>

#include "lbl/config.h"
#include "lbl/factored_nlm.h"
#include "lbl/utils.h"

// Helper functions for reading data, evaluating models, etc.

namespace oxlm {

Real perplexity(
    const boost::shared_ptr<FactoredNLM>& model,
    const boost::shared_ptr<Corpus>& test_corpus);

void saveModel(
    const string& output_file,
    const boost::shared_ptr<FactoredNLM>& model);

void evaluateModel(
    const ModelData& config, const boost::shared_ptr<FactoredNLM>& model,
    const boost::shared_ptr<Corpus>& test_corpus, int minibatch_counter,
    const Time& iteration_start, Real& pp, Real& best_pp);

boost::shared_ptr<FactoredNLM> loadModel(
    const string& input_file,
    const boost::shared_ptr<Corpus>& test_corpus = boost::shared_ptr<Corpus>());

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

} // namespace oxlm
