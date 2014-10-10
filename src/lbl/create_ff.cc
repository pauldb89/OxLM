#include <exception>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include "lbl/cdec_ff_lbl.h"
#include "lbl/cdec_ff_source_lbl.h"

using namespace std;
using namespace oxlm;
namespace po = boost::program_options;

void ParseOptions(
    const string& input, string& filename, string& feature_name,
    oxlm::ModelType& model_type, bool& normalized, bool& persistent_cache) {
  po::options_description options("LBL language model options");
  options.add_options()
      ("file,f", po::value<string>()->required(),
          "File containing serialized language model")
      ("name,n", po::value<string>()->default_value("LBLLM"),
          "Feature name")
      ("type,t", po::value<int>()->required(),
          "Model type")
      ("normalized", po::value<bool>()->required()->default_value(true),
          "Normalize the output of the neural network")
      ("persistent-cache",
          "Cache queries persistently between consecutive decoder runs");

  po::variables_map vm;
  vector<string> args;
  boost::split(args, input, boost::is_any_of(" "));
  po::store(po::command_line_parser(args).options(options).run(), vm);
  po::notify(vm);

  filename = vm["file"].as<string>();
  feature_name = vm["name"].as<string>();
  model_type = static_cast<oxlm::ModelType>(vm["type"].as<int>());
  normalized = vm["normalized"].as<bool>();
  persistent_cache = vm.count("persistent-cache");
}

class UnknownModelException : public exception {
  virtual const char* what() const throw() {
    return "Unknown model type";
  }
};

extern "C" FeatureFunction* create_ff(const string& str) {
  string filename, feature_name;
  oxlm::ModelType model_type;
  bool normalized, persistent_cache;
  ParseOptions(
      str, filename, feature_name, model_type, normalized, persistent_cache);

  switch (model_type) {
    case NLM:
      return new FF_LBLLM<LM>(
          filename, feature_name, normalized, persistent_cache);
    case FACTORED_NLM:
      return new FF_LBLLM<FactoredLM>(
          filename, feature_name, normalized, persistent_cache);
    case FACTORED_MAXENT_NLM:
      return new FF_LBLLM<FactoredMaxentLM>(
          filename, feature_name, normalized, persistent_cache);
    case SOURCE_FACTORED_NLM:
      return new FF_SourceLBLLM(
          filename, feature_name, normalized, persistent_cache);
    default:
      throw UnknownModelException();
  }
}
