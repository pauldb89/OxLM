#include <boost/program_options.hpp>

#include "lbl/factored_metadata.h"
#include "lbl/factored_weights.h"
#include "lbl/model.h"
#include "utils/git_revision.h"

using namespace boost::program_options;
using namespace oxlm;

int main(int argc, char** argv) {
  options_description cmdline_specific("Command line specific options");
  cmdline_specific.add_options()
    ("help,h", "print help message")
    ("config,c", value<string>(),
        "config file specifying additional command line options");

  options_description generic("Allowed options");
  generic.add_options()
    ("input,i", value<string>()->default_value("data.txt"),
        "corpus of sentences, one per line")
    ("test-set", value<string>(),
        "corpus of test sentences to be evaluated at each iteration")
    ("iterations", value<int>()->default_value(10),
        "number of passes through the data")
    ("minibatch-size", value<int>()->default_value(10000),
        "number of sentences per minibatch")
    ("order,n", value<int>()->default_value(4),
        "ngram order")
    ("model-out,o", value<string>(),
        "base filename of model output files")
    ("lambda-lbl,r", value<float>()->default_value(7.0),
        "regularisation strength parameter")
    ("word-width", value<int>()->default_value(100),
        "Width of word representation vectors.")
    ("threads", value<int>()->default_value(1),
        "number of worker threads.")
    ("step-size", value<float>()->default_value(0.05),
        "SGD batch stepsize, it is normalised by the number of minibatches.")
    ("randomise", "visit the training tokens in random order")
    ("diagonal-contexts", "Use diagonal context matrices (usually faster).")
    ("classes", value<int>()->default_value(100),
        "Number of classes for factored output using frequency binning.")
    ("class-file", value<string>(),
        "File containing word to class mappings in the format "
        "<class> <word> <frequence>.");
  options_description config_options, cmdline_options;
  config_options.add(generic);
  cmdline_options.add(generic).add(cmdline_specific);

  variables_map vm;
  store(parse_command_line(argc, argv, cmdline_options), vm);
  if (vm.count("config") > 0) {
    ifstream config(vm["config"].as<string>().c_str());
    store(parse_config_file(config, cmdline_options), vm);
  }
  notify(vm);

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
  config.ngram_order = vm["order"].as<int>();

  if (vm.count("model-in")) {
    config.model_input_file = vm["model-in"].as<string>();
  }
  if (vm.count("model-out")) {
    config.model_output_file = vm["model-out"].as<string>();
    if (GIT_REVISION) {
      config.model_output_file += "." + string(GIT_REVISION);
    }
  }

  config.l2_lbl = vm["lambda-lbl"].as<float>();
  config.word_representation_size = vm["word-width"].as<int>();
  config.threads = vm["threads"].as<int>();
  config.step_size = vm["step-size"].as<float>();
  config.randomise = vm.count("randomise");
  config.diagonal_contexts = vm.count("diagonal-contexts");

  config.classes = vm["classes"].as<int>();
  if (vm.count("class-file")) {
    config.class_file = vm["class-file"].as<string>();
  }

  cout << "################################" << endl;
  if (strlen(GIT_REVISION) > 0) {
    cout << "# Git revision: " << GIT_REVISION << endl;
  }
  cout << "# Config Summary" << endl;
  cout << "# order = " << config.ngram_order << endl;
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
  cout << "# lambda = " << config.l2_lbl << endl;
  cout << "# step size = " << config.step_size << endl;
  cout << "# iterations = " << config.iterations << endl;
  cout << "# threads = " << config.threads << endl;
  cout << "################################" << endl;

  Model<FactoredWeights, FactoredWeights, FactoredMetadata> model(config);
  model.learn();

  return 0;
}
