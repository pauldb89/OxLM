#include <fstream>
#include <string>

#include <boost/program_options.hpp>

#include "lbl/model.h"

using namespace boost::program_options;
using namespace oxlm;
using namespace std;

template<class Model>
void ExtractWordVectors(
    const string& model_file,
    const string& vocab_file,
    const string& vectors_file) {
  Model model;
  model.load(model_file);

  Dict dict = model.getDict();
  MatrixReal word_vectors = model.getWordVectors();

  ofstream fout(vocab_file);
  for (size_t i = 0; i < word_vectors.cols(); ++i) {
    fout << dict.Convert(i) << endl;
  }


  ofstream vout(vectors_file);
  for (size_t i = 0; i < word_vectors.cols(); ++i) {
    vout << word_vectors.col(i).transpose() << endl;
  }
}

int main(int argc, char** argv) {
  options_description desc("Command line options");
  desc.add_options()
      ("help,h", "Print available options")
      ("model,m", value<string>()->required(), "File containing the model")
      ("type,t", value<int>()->required(), "Model type")
      ("vocab", value<string>()->required(), "Output file for model vocabulary")
      ("vectors", value<string>()->required(), "Output file for word vectors");

  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    cout << desc << endl;
  }

  notify(vm);

  string model_file = vm["model"].as<string>();
  int model_type = vm["type"].as<int>();
  string vocab_file = vm["vocab"].as<string>();
  string vectors_file = vm["vectors"].as<string>();

  switch (model_type) {
    case NLM:
      ExtractWordVectors<LM>(model_file, vocab_file, vectors_file);
      return 0;
    case FACTORED_NLM:
      ExtractWordVectors<FactoredLM>(model_file, vocab_file, vectors_file);
      return 0;
    case FACTORED_MAXENT_NLM:
      ExtractWordVectors<FactoredMaxentLM>(model_file, vocab_file, vectors_file);
      return 0;
    default:
      cout << "Unknown model type" << endl;
      return 1;
  }

  return 0;
}
