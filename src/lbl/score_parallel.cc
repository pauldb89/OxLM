#include <boost/program_options.hpp>

#include "lbl/model.h"
#include "lbl/parallel_processor.h"

using namespace boost::program_options;
using namespace std;
using namespace oxlm;

int main(int argc, char** argv) {
  options_description desc("Command line options");
  desc.add_options()
      ("help,h", "Print help message")
      ("model,m", value<string>()->required(), "File containing the model.")
      ("test,t", value<string>()->required(),
          "File containing parallel test corpus.")
      ("alignment,a", value<string>()->required(),
          "File containing the alignments");

  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    cout << desc << endl;
    return 0;
  }

  notify(vm);

  SourceFactoredLM model;
  model.load(vm["model"].as<string>());
  boost::shared_ptr<Vocabulary> vocab = model.getVocab();
  boost::shared_ptr<ModelData> config = model.getConfig();
  config->test_file = vm["test"].as<string>();
  config->test_alignment_file = vm["alignment"].as<string>();

  boost::shared_ptr<Corpus> test_corpus =
      readTestCorpus(config, vocab, true);

  int context_width = config->ngram_order - 1;
  int source_context_width = 2 * config->source_order - 1;
  ParallelProcessor processor(test_corpus, context_width, source_context_width);

  double total = 0;
  int eos = vocab->convert("</s>");
  for (size_t i = 0; i < test_corpus->size(); ++i) {
    int word_id = test_corpus->at(i);
    vector<int> context = processor.extract(i);
    double log_prob = model.predict(word_id, context);
    total += log_prob;
    cout << "(" << vocab->convert(word_id) << " " << log_prob << ") ";
    if (word_id == eos) {
      cout << "Total: " << total << endl;
      total = 0;
    }
  }

  return 0;
}
