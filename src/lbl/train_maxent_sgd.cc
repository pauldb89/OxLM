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
    ("log-period", value<int>()->default_value(0),
        "Log model every X iterations")
    ("lambda,r", value<float>()->default_value(7.0),
        "regularisation strength parameter")
    ("word-width", value<int>()->default_value(100),
        "Width of word representation vectors.")
    ("threads", value<int>()->default_value(1),
        "number of worker threads.")
    ("step-size", value<float>()->default_value(0.05),
        "SGD batch stepsize, it is normalised by the number of minibatches.")
    ("classes", value<int>()->default_value(100),
        "number of classes for factored output.")
    ("class-file", value<string>(),
        "file containing word to class mappings in the format <class> <word> <frequence>.")
    ("randomise", "visit the training tokens in random order")
    ("reclass", "reallocate word classes after the first epoch.")
    ("diagonal-contexts", "Use diagonal context matrices (usually faster).")
    ("sparse-features", value<bool>()->default_value(true),
        "Only define maximum entropy feature functions for observed contexts")
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
  }
  config.log_period = vm["log-period"].as<int>();
  config.l2_parameter = vm["lambda"].as<float>();
  config.word_representation_size = vm["word-width"].as<int>();
  config.threads = vm["threads"].as<int>();
  config.step_size = vm["step-size"].as<float>();
  config.classes = vm["classes"].as<int>();
  config.class_file = vm["class-file"].as<string>();
  config.randomise = vm.count("randomise");
  config.reclass = vm.count("reclass");
  config.diagonal_contexts = vm.count("diagonal-contexts");
  config.sparse_features = vm["sparse-features"].as<bool>();
  config.random_weights = vm["random-weights"].as<bool>();

  cerr << "################################" << endl;
  cerr << "# Config Summary" << endl;
  cerr << "# order = " << config.ngram_order << endl;
  cerr << "# feature-context-size = " << config.feature_context_size;
  if (config.model_input_file.size()) {
    cerr << "# model-in = " << config.model_input_file << endl;
  }
  if (config.model_output_file.size()) {
    cerr << "# model-out = " << config.model_output_file << endl;
  }
  cerr << "# input = " << config.training_file << endl;
  cerr << "# minibatch-size = " << config.minibatch_size << endl;
  cerr << "# lambda = " << config.l2_parameter << endl;
  cerr << "# step size = " << config.step_size << endl;
  cerr << "# iterations = " << config.iterations << endl;
  cerr << "# threads = " << config.threads << endl;
  cerr << "# classes = " << config.classes << endl;
  cerr << "################################" << endl;

  learn(config);

  return 0;
}
