#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>

#include "lbl/model.h"

using namespace boost::program_options;
using namespace oxlm;

template<class Model>
void predict(const string& model_file, const string& contexts_file) {
  Model model;
  model.load(model_file);

  Dict dict = model.getDict();
  int kUNKNOWN = dict.Convert("<unk>");

  string line;
  ifstream in(contexts_file);
  while (getline(in, line)) {
    istringstream sin(line);
    string word;
    vector<int> context;
    while (sin >> word) {
      context.push_back(dict.Convert(word));
    }

    int context_length = context.size();
    // Context need to be reversed.
    // (i.e. in the ukrainian parliament => parliament ukrainian the in)
    reverse(context.begin(), context.end());
    context.resize(4, kUNKNOWN);

    vector<pair<double, int>> outcomes;
    for (int word_id = 0; word_id < dict.size(); ++word_id) {
      outcomes.push_back(make_pair(
          exp(model.predict(word_id, context)), word_id));
    }

    sort(outcomes.begin(), outcomes.end());

    double sum = 0;
    for (const auto& outcome: outcomes) {
      for (int i = context_length - 1; i >= 0; --i) {
        cout << dict.Convert(context[i]) << " ";
      }
      cout << dict.Convert(outcome.second) << " " << outcome.first << endl;
      sum += outcome.first;
    }

    cout << "sum: " << sum << endl;
    cout << "====================" << endl;
    assert(fabs(1 - sum) < 1e-5);
  }


}

int main(int argc, char** argv) {
  options_description desc("General options");
  desc.add_options()
      ("help,h", "Print help message")
      ("model,m", value<string>()->required(), "File containing the model")
      ("type,t", value<int>()->required(), "Model type")
      ("contexts,c", value<string>()->required(),
          "File containing the contexts");

  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    cout << desc << endl;
    return 0;
  }

  notify(vm);

  string model_file = vm["model"].as<string>();
  string contexts_file = vm["contexts"].as<string>();
  ModelType model_type = static_cast<ModelType>(vm["type"].as<int>());

  switch (model_type) {
    case NLM:
      predict<LM>(model_file, contexts_file);
      return 0;
    case FACTORED_NLM:
      predict<FactoredLM>(model_file, contexts_file);
      return 0;
    case FACTORED_MAXENT_NLM:
      predict<FactoredMaxentLM>(model_file, contexts_file);
      return 0;
    default:
      cout << "Unknown model type" << endl;
      return 1;
  }

  return 0;
}
