#include <exception>
#include <iostream>
#include <string>

#include <boost/algorithm/string.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>

#include "corpus/corpus.h"
#include "lbl/cdec_lbl_mapper.h"
#include "lbl/cdec_rule_converter.h"
#include "lbl/cdec_state_converter.h"
#include "lbl/lbl_features.h"
#include "lbl/model.h"
#include "lbl/process_identifier.h"
#include "lbl/query_cache.h"

// cdec headers
#include "ff.h"
#include "hg.h"
#include "sentence_metadata.h"

using namespace std;
using namespace oxlm;
namespace po = boost::program_options;

namespace oxlm {

template<class Model>
class FF_LBLLM : public FeatureFunction {
 public:
  FF_LBLLM(
      const string& filename, const string& feature_name,
      bool cache_queries)
      : fid(FD::Convert(feature_name)),
        fidOOV(FD::Convert(feature_name + "_OOV")),
        processIdentifier("FF_LBLLM"),
        cacheQueries(cache_queries), cacheHits(0), totalHits(0) {
    model.load(filename);

    // Note: This is a hack due to lack of time.
    // Ideally, we would like a to have client server architecture, where the
    // server contains both the LM and the n-gram cache, preventing these huge
    // data structures from being replicated to every process. Also, that
    // approach would not require us to save the n-gram cache to disk after
    // every MIRA iteration.
    if (cacheQueries) {
      processId = processIdentifier.reserveId();
      cerr << "Reserved id " << processId
           << " at time " << Clock::to_time_t(GetTime()) << endl;
      cacheFile = filename + "." + to_string(processId) + ".cache.bin";
      if (boost::filesystem::exists(cacheFile)) {
        ifstream f(cacheFile);
        boost::archive::binary_iarchive ia(f);
        cerr << "Loading n-gram probability cache from " << cacheFile << endl;
        ia >> cache;
        cerr << "Finished loading " << cache.size()
             << " n-gram probabilities..." << endl;
      } else {
        cerr << "Cache file not found..." << endl;
      }
    }

    config = model.getConfig();
    int context_width = config.ngram_order - 1;
    // For each state, we store at most context_width word ids to the left and
    // to the right and a kSTAR separator. The last bit represents the actual
    // size of the state.
    int max_state_size = (2 * context_width + 1) * sizeof(int) + 1;
    FeatureFunction::SetStateSize(max_state_size);

    dict = model.getDict();
    mapper = boost::make_shared<CdecLBLMapper>(dict);
    stateConverter = boost::make_shared<CdecStateConverter>(max_state_size - 1);
    ruleConverter = boost::make_shared<CdecRuleConverter>(mapper, stateConverter);

    kSTART = dict.Convert("<s>");
    kSTOP = dict.Convert("</s>");
    kUNKNOWN = dict.Convert("<unk>");
    kSTAR = dict.Convert("<{STAR}>");
  }

  virtual void PrepareForInput(const SentenceMetadata& smeta) {
    model.clearCache();
  }

 protected:
  virtual void TraversalFeaturesImpl(
      const SentenceMetadata& smeta, const HG::Edge& edge,
      const vector<const void*>& prev_states, SparseVector<double>* features,
      SparseVector<double>* estimated_features, void* next_state) const {
    vector<int> symbols = ruleConverter->convertTargetSide(
        edge.rule_->e(), prev_states);

    LBLFeatures exact_scores = scoreFullContexts(symbols);
    if (exact_scores.LMScore) {
      features->set_value(fid, exact_scores.LMScore);
    }
    if (exact_scores.OOVScore) {
      features->set_value(fidOOV, exact_scores.OOVScore);
    }

    constructNextState(symbols, next_state);
    symbols = stateConverter->convert(next_state);

    LBLFeatures estimated_scores = estimateScore(symbols);
    if (estimated_scores.LMScore) {
      estimated_features->set_value(fid, estimated_scores.LMScore);
    }
    if (estimated_scores.OOVScore) {
      estimated_features->set_value(fidOOV, estimated_scores.OOVScore);
    }
  }

  virtual void FinalTraversalFeatures(
      const void* prev_state, SparseVector<double>* features) const {
    vector<int> symbols = stateConverter->convert(prev_state);
    symbols.insert(symbols.begin(), kSTART);
    symbols.push_back(kSTOP);

    LBLFeatures final_scores = estimateScore(symbols);
    if (final_scores.LMScore) {
      features->set_value(fid, final_scores.LMScore);
    }
    if (final_scores.OOVScore) {
      features->set_value(fidOOV, final_scores.OOVScore);
    }
  }

 private:
  LBLFeatures scoreFullContexts(const vector<int>& symbols) const {
    LBLFeatures ret;
    int last_star = -1;
    int context_width = config.ngram_order - 1;
    for (size_t i = 0; i < symbols.size(); ++i) {
      if (symbols[i] == kSTAR) {
        last_star = i;
      } else if (i - last_star > context_width) {
        ret += scoreContext(symbols, i);
      }
    }

    return ret;
  }

  LBLFeatures scoreContext(const vector<int>& symbols, int position) const {
    int word = symbols[position];
    int context_width = config.ngram_order - 1;
    vector<int> context;
    for (int i = 1; i <= context_width && position - i >= 0; ++i) {
      assert(symbols[position - i] != kSTAR);
      context.push_back(symbols[position - i]);
    }

    if (!context.empty() && context.back() == kSTART) {
      context.resize(context_width, kSTART);
    } else {
      context.resize(context_width, kUNKNOWN);
    }

    double score;
    if (!cacheQueries) {
      score = model.predict(word, context);
    } else {
      NGram query(word, context);
      ++totalHits;
      pair<double, bool> ret = cache.get(query);
      if (ret.second) {
        ++cacheHits;
        score = ret.first;
      } else {
        score = model.predict(word, context);
        cache.put(query, score);
      }
    }

    return LBLFeatures(score, word == kUNKNOWN);
  }

  void constructNextState(const vector<int>& symbols, void* state) const {
    int context_width = config.ngram_order - 1;

    vector<int> next_state;
    for (size_t i = 0; i < symbols.size() && i < context_width; ++i) {
      if (symbols[i] == kSTAR) {
        break;
      }
      next_state.push_back(symbols[i]);
    }

    if (next_state.size() < symbols.size()) {
      next_state.push_back(kSTAR);

      int last_star = -1;
      for (size_t i = 0; i < symbols.size(); ++i) {
        if (symbols[i] == kSTAR) {
          last_star = i;
        }
      }

      size_t i = max(last_star + 1, static_cast<int>(symbols.size() - context_width));
      while (i < symbols.size()) {
        next_state.push_back(symbols[i]);
        ++i;
      }
    }

    stateConverter->convert(next_state, state);
  }

  LBLFeatures estimateScore(const vector<int>& symbols) const {
    LBLFeatures ret = scoreFullContexts(symbols);

    int context_width = config.ngram_order - 1;
    for (size_t i = 0; i < symbols.size() && i < context_width; ++i) {
      if (symbols[i] == kSTAR) {
        break;
      }

      if (symbols[i] != kSTART) {
        ret += scoreContext(symbols, i);
      }
    }

    return ret;
  }

  int fid;
  int fidOOV;
  Dict dict;
  ModelData config;
  Model model;
  boost::shared_ptr<CdecLBLMapper> mapper;
  boost::shared_ptr<CdecRuleConverter> ruleConverter;
  boost::shared_ptr<CdecStateConverter> stateConverter;

  int kSTART;
  int kSTOP;
  int kUNKNOWN;
  int kSTAR;

  ProcessIdentifier processIdentifier;
  int processId;

  bool cacheQueries;
  string cacheFile;
  mutable QueryCache cache;
  mutable int cacheHits, totalHits;
};

} // namespace oxlm

void ParseOptions(
    const string& input, string& filename, string& feature_name,
    oxlm::ModelType& model_type, bool& cache_queries) {
  po::options_description options("LBL language model options");
  options.add_options()
      ("file,f", po::value<string>()->required(),
          "File containing serialized language model")
      ("name,n", po::value<string>()->default_value("LBLLM"),
          "Feature name")
      ("type,t", po::value<int>()->required(),
          "Model type")
      ("cache-queries", "Whether should cache n-gram probabilities.");

  po::variables_map vm;
  vector<string> args;
  boost::split(args, input, boost::is_any_of(" "));
  po::store(po::command_line_parser(args).options(options).run(), vm);
  po::notify(vm);

  filename = vm["file"].as<string>();
  feature_name = vm["name"].as<string>();
  model_type = static_cast<oxlm::ModelType>(vm["type"].as<int>());
  cache_queries = vm.count("cache-queries");
}

class UnknownModelException : public exception {
  virtual const char* what() const throw() {
    return "Unknown model type";
  }
};

extern "C" FeatureFunction* create_ff(const string& str) {
  string filename, feature_name;
  oxlm::ModelType model_type;
  bool cache_queries;

  ParseOptions(str, filename, feature_name, model_type, cache_queries);

  switch (model_type) {
    case NLM:
      return new FF_LBLLM<LM>(filename, feature_name, cache_queries);
    case FACTORED_NLM:
      return new FF_LBLLM<FactoredLM>(filename, feature_name, cache_queries);
    case FACTORED_MAXENT_NLM:
      return new FF_LBLLM<FactoredMaxentLM>(filename, feature_name, cache_queries);
    default:
      throw UnknownModelException();
  }
}
