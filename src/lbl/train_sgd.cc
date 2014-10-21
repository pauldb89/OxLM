#include <boost/program_options.hpp>

#include "lbl/metadata.h"
#include "lbl/model.h"
#include "lbl/weights.h"
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
        "Maximum number of passes through the data.")
    ("evaluate-frequency", value<int>()->default_value(1000),
        "Evaluate models every N minibatches")
    ("minibatch-size", value<int>()->default_value(10000),
        "number of sentences per minibatch")
    ("minibatch-threshold", value<int>()->default_value(20000),
        "Stop training if the test perplexity did not improve in the last "
        "N minibatches. Note that test perplexities are calculated only "
        "every evaluate-frequency minibatches")
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
    ("randomise", value<bool>()->default_value(true),
        "Visit the training tokens in random order.")
    ("diagonal-contexts", value<bool>()->default_value(true),
        "Use diagonal context matrices (usually faster).")
    ("activation", value<int>()->default_value(2),
        "Activation function for the prediction (hidden) layer. "
        "0: Identity, 1: Sigmoid, 2: Rectifier.")
    ("noise-samples", value<int>()->default_value(0),
        "Number of noise samples for noise contrastive estimation. "
        "If zero, minibatch gradient descent is used instead.");
  options_description config_options, cmdline_options;
  config_options.add(generic);
  cmdline_options.add(generic).add(cmdline_specific);

  variables_map vm;
  store(parse_command_line(argc, argv, cmdline_options), vm);
  if (vm.count("config") > 0) {
    ifstream config(vm["config"].as<string>().c_str());
    store(parse_config_file(config, cmdline_options), vm);
  }

  if (vm.count("help")) {
    cout << cmdline_options << "\n";
    return 1;
  }

  notify(vm);

  boost::shared_ptr<ModelData> config = boost::make_shared<ModelData>();
  config->training_file = vm["input"].as<string>();
  if (vm.count("test-set")) {
    config->test_file = vm["test-set"].as<string>();
  }
  config->iterations = vm["iterations"].as<int>();
  config->evaluate_frequency = vm["evaluate-frequency"].as<int>();
  config->minibatch_size = vm["minibatch-size"].as<int>();
  config->minibatch_threshold = vm["minibatch-threshold"].as<int>();
  config->ngram_order = vm["order"].as<int>();

  if (vm.count("model-out")) {
    config->model_output_file = vm["model-out"].as<string>();
    if (GIT_REVISION) {
      config->model_output_file += "." + string(GIT_REVISION);
    }
  }

  config->l2_lbl = vm["lambda-lbl"].as<float>();
  config->word_representation_size = vm["word-width"].as<int>();
  config->threads = vm["threads"].as<int>();
  config->step_size = vm["step-size"].as<float>();
  config->randomise = vm["randomise"].as<bool>();
  config->diagonal_contexts = vm["diagonal-contexts"].as<bool>();
  config->activation = static_cast<Activation>(vm["activation"].as<int>());

  config->noise_samples = vm["noise-samples"].as<int>();

  cout << "################################" << endl;
  if (strlen(GIT_REVISION) > 0) {
    cout << "# Git revision: " << GIT_REVISION << endl;
  }
  cout << "# Config Summary" << endl;
  cout << "# order = " << config->ngram_order << endl;
  cout << "# word_width = " << config->word_representation_size << endl;
  if (config->model_output_file.size()) {
    cout << "# model-out = " << config->model_output_file << endl;
  }
  cout << "# input = " << config->training_file << endl;
  cout << "# minibatch size = " << config->minibatch_size << endl;
  cout << "# minibatch threshold = " << config->minibatch_threshold << endl;
  cout << "# lambda = " << config->l2_lbl << endl;
  cout << "# step size = " << config->step_size << endl;
  cout << "# iterations = " << config->iterations << endl;
  cout << "# evaluate frequency = " << config->evaluate_frequency << endl;
  cout << "# threads = " << config->threads << endl;
  cout << "# randomise = " << config->randomise << endl;
  cout << "# diagonal contexts = " << config->diagonal_contexts << endl;
  cout << "# activation = " << config->activation << endl;
  cout << "# noise samples = " << config->noise_samples << endl;
  cout << "################################" << endl;

  Model<Weights, Weights, Metadata> model(config);
  model.learn();

  return 0;
}
