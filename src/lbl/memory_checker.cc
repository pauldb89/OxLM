#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include "lbl/model.h"

using namespace boost::program_options;
using namespace oxlm;
using namespace std;


template<class Model>
void loadModel(const string& model_file) {
  Model model;
  model.load(model_file);

  printMemoryUsage();
}


int main(int argc, char** argv) {
  options_description desc("Command line options");
  desc.add_options()
    ("help,h", "Print help message.")
    ("model,m", value<string>()->required(), "File containing the model")
    ("type,t", value<int>()->required(), "Model type");

  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    cout << desc << endl;
    return 0;
  }

  notify(vm);

  string model_file = vm["model"].as<string>();
  ModelType model_type = static_cast<ModelType>(vm["type"].as<int>());

  switch (model_type) {
    case NLM:
      loadModel<LM>(model_file);
      return 0;
    case FACTORED_NLM:
      loadModel<FactoredLM>(model_file);
      return 0;
    case FACTORED_MAXENT_NLM:
      loadModel<FactoredMaxentLM>(model_file);
      return 0;
    case FACTORED_TREE_NLM:
      loadModel<FactoredTreeLM>(model_file);
      return 0;
    default:
      cout << "Unknown model type" << endl;
      return 1;
  }

  return 0;
}
