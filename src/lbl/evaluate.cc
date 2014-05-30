#include <boost/program_options.hpp>

#include "corpus/corpus.h"
#include "lbl/context_processor.h"
#include "lbl/factored_nlm.h"
#include "lbl/model_utils.h"
#include "lbl/utils.h"

using namespace boost::program_options;
using namespace oxlm;
using namespace std;

int main(int argc, char** argv) {
  options_description desc("Command line options");
  desc.add_options()
      ("help,h", "Print help message.")
      ("test,t", value<string>()->required(), "File containing the test corpus")
      ("model,m", value<string>()->required(), "File containing the model");

  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);

  boost::shared_ptr<FactoredNLM> model = loadModel(vm["model"].as<string>());

  Dict dict = model->label_set();
  int eos = dict.Convert("</s>");
  boost::shared_ptr<Corpus> test_corpus =
      readCorpus(vm["test"].as<string>(), dict, true);

  double total = 0;
  ModelData config = model->config;
  ContextProcessor processor(test_corpus, config.ngram_order - 1);
  for (size_t i = 0; i < test_corpus->size(); ++i) {
    int word_id = test_corpus->at(i);
    vector<int> context = processor.extract(i);
    double log_prob = model->log_prob(word_id, context, true, true);
    total += log_prob;
    cout << "(" << dict.Convert(word_id) << " " << log_prob << ") ";
    if (word_id == eos) {
      cout << "Total: " << total << endl;
      total = 0;
    }
  }

  return 0;
}
