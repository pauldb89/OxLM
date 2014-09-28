#include <boost/program_options.hpp>

#include "lbl/model.h"

using namespace boost::program_options;
using namespace oxlm;
using namespace std;

int main(int argc, char** argv) {
  options_description desc("Command line options");
  desc.add_options()
      ("help,h", "Print help message.")
      ("model,m", value<string>()->required(), "File containing the model")
      ("test,t", value<string>()->required(),
          "File containing the parallel test corpus")
      ("alignment,a", value<string>()->required(),
          "File containing the word alignments")
      ("threads", value<int>()->required()->default_value(1),
          "Number of threads for evaluation.");

  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    cout << desc << endl;
    return 0;
  }

  notify(vm);

  SourceFactoredLM model;
  model.load(vm["model"].as<string>());
  boost::shared_ptr<ModelData> config = model.getConfig();

  config->test_file = vm["test"].as<string>();
  config->test_alignment_file = vm["alignment"].as<string>();

  boost::shared_ptr<Vocabulary> vocab = model.getVocab();
  boost::shared_ptr<Corpus> test_corpus =
      readTestCorpus(config, vocab, true);

  Real accumulator = 0;
  int num_threads = vm["threads"].as<int>();
  #pragma omp parallel num_threads(num_threads)
  model.evaluate(test_corpus, accumulator);

  cout << "Test set perplexity: "
       << perplexity(accumulator, test_corpus->size()) << endl;

  return 0;
}
