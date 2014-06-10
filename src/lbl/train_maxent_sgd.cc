#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include "lbl/train_maxent_sgd.h"

static const char *REVISION = "$Rev: 247 $";

// Namespaces
using namespace boost;
using namespace boost::program_options;
using namespace std;
using namespace oxlm;
using namespace Eigen;


int main(int argc, char **argv) {
  cout << "Online noise contrastive estimation for log-bilinear models: Copyright 2013 Phil Blunsom, "
       << REVISION << '\n' << endl;

  ///////////////////////////////////////////////////////////////////////////////////////
  // Command line processing
  variables_map vm;

  // Command line processing
  options_description cmdline_specific("Command line specific options");
  cmdline_specific.add_options()
    ("help,h", "print help message")
    ("config,c", value<string>(),
        "config file specifying additional command line options")
    ;
  options_description generic("Allowed options");
  generic.add_options()
    ("input,i", value<string>()->default_value("data.txt"),
        "corpus of sentences, one per line")
    ("test-set", value<string>(),
        "corpus of test sentences to be evaluated at each iteration")
    ("iterations", value<int>()->default_value(10),
        "number of passes through the data")
    ("minibatch-size", value<int>()->default_value(100),
        "number of sentences per minibatch")
    ("instances", value<int>()->default_value(std::numeric_limits<int>::max()),
        "training instances per iteration")
    ("order,n", value<int>()->default_value(5),
        "ngram order")
    ("feature-context-size", value<int>()->default_value(5),
        "size of the window for maximum entropy features")
    ("model-in,m", value<string>(),
        "initial model")
    ("model-out,o", value<string>(),
        "base filename of model output files")
    ("lambda-lbl,r", value<float>()->default_value(2.0),
        "LBL regularisation strength parameter")
    ("lambda-maxent", value<float>()->default_value(2.0),
        "maxent regularisation strength parameter")
    ("word-width", value<int>()->default_value(100),
        "Width of word representation vectors.")
    ("threads", value<int>()->default_value(1),
        "number of worker threads.")
    ("step-size", value<float>()->default_value(0.05),
        "SGD batch stepsize, it is normalised by the number of minibatches.")
    ("classes", value<int>()->default_value(100),
        "number of classes for factored output.")
    ("class-file", value<string>(),
        "file containing word to class mappings in the format"
        "<class> <word> <frequence>.")
    ("randomise", "visit the training tokens in random order")
    ("reclass", "reallocate word classes after the first epoch.")
    ("diagonal-contexts", "Use diagonal context matrices (usually faster).")
    ("max-ngrams", value<int>()->default_value(0),
        "Define maxent features only for the most frequent max-ngrams ngrams.")
    ("min-ngram-freq", value<int>()->default_value(1),
        "Define maxent features only for n-grams above this frequency.")
    ("sparse-features", value<bool>()->default_value(true),
        "Only define maximum entropy feature functions for observed contexts")
    ("hash-space", value<Real>()->default_value(0),
        "The size of the space in which the maxent features are mapped to "
        "(in millions).")
    ("filter-contexts", value<bool>()->default_value(true),
        "Prevent false contexts from being hashed.")
    ("filter-error-rate", value<Real>()->default_value(0),
        "Error rate for filtering false contexts (in bloom filter)")
    ("count-collisions", value<bool>()->default_value(true),
        "Print collision statistics (leads to a memory usage spike)")
    ("random-weights", value<bool>()->default_value(true),
        "Initialize the weights randomly.");
  options_description config_options, cmdline_options;
  config_options.add(generic);
  cmdline_options.add(generic).add(cmdline_specific);

  store(parse_command_line(argc, argv, cmdline_options), vm);
  if (vm.count("config") > 0) {
    ifstream config(vm["config"].as<string>().c_str());
    store(parse_config_file(config, cmdline_options), vm);
  }
  notify(vm);
  ///////////////////////////////////////////////////////////////////////////////////////

  if (vm.count("help")) {
    cout << cmdline_options << "\n";
    return 1;
  }

  ModelData config;
  config.training_file = vm["input"].as<string>();
  if (vm.count("test-set")) {
    config.test_file = vm["test-set"].as<string>();
  }
  config.iterations = vm["iterations"].as<int>();
  config.minibatch_size = vm["minibatch-size"].as<int>();
  config.instances = vm["instances"].as<int>();
  config.ngram_order = vm["order"].as<int>();
  config.feature_context_size = vm["feature-context-size"].as<int>();

  if (vm.count("model-in")) {
    config.model_input_file = vm["model-in"].as<string>();
  }
  if (vm.count("model-out")) {
    config.model_output_file = vm["model-out"].as<string>();
    if (GIT_REVISION) {
      config.model_output_file += "." + string(GIT_REVISION);
    }
  }
  if (vm.count("class-file")) {
    config.class_file = vm["class-file"].as<string>();
  }

  config.l2_lbl = vm["lambda-lbl"].as<float>();
  config.l2_maxent = vm["lambda-maxent"].as<float>();
  config.word_representation_size = vm["word-width"].as<int>();
  config.threads = vm["threads"].as<int>();
  config.step_size = vm["step-size"].as<float>();
  config.classes = vm["classes"].as<int>();
  config.randomise = vm.count("randomise");
  config.reclass = vm.count("reclass");
  config.diagonal_contexts = vm.count("diagonal-contexts");
  config.random_weights = vm["random-weights"].as<bool>();

  config.max_ngrams = vm["max-ngrams"].as<int>();
  config.min_ngram_freq = vm["min-ngram-freq"].as<int>();

  config.sparse_features = vm["sparse-features"].as<bool>();
  config.hash_space = vm["hash-space"].as<Real>() * 1000000;
  config.filter_contexts = vm["filter-contexts"].as<bool>();
  config.filter_error_rate = vm["filter-error-rate"].as<Real>();
  config.count_collisions = vm["count-collisions"].as<bool>();

  cout << "################################" << endl;
  if (strlen(GIT_REVISION) > 0) {
    cout << "# Git revision: " << GIT_REVISION << endl;
  }
  cout << "# Config Summary" << endl;
  cout << "# order = " << config.ngram_order << endl;
  cout << "# feature context size = " << config.feature_context_size << endl;
  cout << "# word_width = " << config.word_representation_size << endl;
  cout << "# diagonal contexts = " << config.diagonal_contexts << endl;
  if (config.model_input_file.size()) {
    cout << "# model-in = " << config.model_input_file << endl;
  }
  if (config.model_output_file.size()) {
    cout << "# model-out = " << config.model_output_file << endl;
  }
  cout << "# input = " << config.training_file << endl;
  cout << "# minibatch size = " << config.minibatch_size << endl;
  cout << "# lambda LBL = " << config.l2_lbl << endl;
  cout << "# lambda maxent = " << config.l2_maxent << endl;
  cout << "# step size = " << config.step_size << endl;
  cout << "# iterations = " << config.iterations << endl;
  cout << "# threads = " << config.threads << endl;
  cout << "# classes = " << config.classes << endl;
  cout << "# max n-grams = " << config.max_ngrams << endl;
  cout << "# min n-gram frequency = " << config.min_ngram_freq << endl;
  cout << "# sparse features = " << config.sparse_features << endl;
  cout << "# hash space = " << config.hash_space << endl;
  cout << "# filter contexts = " << config.filter_contexts << endl;
  cout << "# filter error rate = " << config.filter_error_rate << endl;
  cout << "################################" << endl;

  learn(config);

  return 0;
}
