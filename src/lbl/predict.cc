#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>

#include "lbl/factored_nlm.h"
#include "lbl/model_utils.h"

using namespace boost::program_options;
using namespace oxlm;

int main(int argc, char** argv) {
  options_description desc("General options");
  desc.add_options()
      ("help,h", "Print help message")
      ("model,m", value<string>()->required(), "File containing the model")
      ("contexts,c", value<string>()->required(),
          "File containing the contexts");

  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    cout << desc << endl;
    return 0;
  }

  notify(vm);

  boost::shared_ptr<FactoredNLM> model = loadModel(
      vm["model"].as<string>(), boost::shared_ptr<Corpus>());

  int kUNKNOWN = model->label_id("<unk>");

  string line;
  ifstream in(vm["contexts"].as<string>());
  while (getline(in, line)) {
    istringstream sin(line);
    string word;
    vector<int> context;
    while (sin >> word) {
      context.push_back(model->label_id(word));
    }

    int context_length = context.size();
    // Context need to be reversed.
    // (i.e. in the ukrainian parliament => parliament ukrainian the in)
    reverse(context.begin(), context.end());
    context.resize(4, kUNKNOWN);

    vector<pair<double, int>> outcomes;
    for (int word_id = 0; word_id < model->labels(); ++word_id) {
      outcomes.push_back(make_pair(
          exp(model->log_prob(word_id, context, true, true)), word_id));
    }

    sort(outcomes.begin(), outcomes.end());

    double sum = 0;
    for (const auto& outcome: outcomes) {
      for (int i = context_length - 1; i >= 0; --i) {
        cout << model->label_str(context[i]) << " ";
      }
      cout << model->label_str(outcome.second) << " " << outcome.first << endl;
      sum += outcome.first;
    }

    cout << "sum: " << sum << endl;
    cout << "====================" << endl;
    assert(fabs(1 - sum) < 1e-5);
  }

  return 0;
}
