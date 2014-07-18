#include <iostream>

#include <boost/program_options.hpp>

#include "lbl/model.h"
#include "lbl/model_utils.h"
#include "lbl/utils.h"

using namespace boost::program_options;
using namespace oxlm;
using namespace std;

template<class Model>
void evaluate(const string& model_file, const string& test_file, int num_threads) {
  Model model;
  model.load(model_file);
  Dict dict = model.getDict();
  boost::shared_ptr<Corpus> test_corpus = readCorpus(test_file, dict, true, true);

  Real accumulator = 0;
  #pragma omp parallel num_threads(num_threads)
  model.evaluate(test_corpus, accumulator);

  cout << "Test set perplexity: "
       << perplexity(accumulator, test_corpus->size()) << endl;
}

int main(int argc, char** argv) {
  options_description desc("Command line options");
  desc.add_options()
      ("help,h", "Print help message.")
      ("model,m", value<string>()->required(), "File containing the model")
      ("type,t", value<int>()->required(), "Model type")
      ("data,d", value<string>()->required(), "File containing the test set.")
      ("threads", value<int>()->required()->default_value(1),
          "Number of threads for evaluation.");

  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    cout << desc << endl;
    return 0;
  }

  notify(vm);

  string model_file = vm["model"].as<string>();
  string test_file = vm["data"].as<string>();
  int num_threads = vm["threads"].as<int>();
  ModelType model_type = static_cast<ModelType>(vm["type"].as<int>());

  switch (model_type) {
    case NLM:
      evaluate<LM>(model_file, test_file, num_threads);
      return 0;
    case FACTORED_NLM:
      evaluate<FactoredLM>(model_file, test_file, num_threads);
      return 0;
    case FACTORED_MAXENT_NLM:
      evaluate<FactoredMaxentLM>(model_file, test_file, num_threads);
      return 0;
    default:
      cout << "Unknown model type" << endl;
      return 1;
  }

  return 0;
}
