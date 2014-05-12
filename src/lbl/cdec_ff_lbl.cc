#include <iostream>
#include <string>

#include <boost/algorithm/string.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/shared_ptr.hpp>

#include "corpus/corpus.h"
#include "lbl/factored_nlm.h"
#include "lbl/factored_maxent_nlm.h"
#include "lbl/process_identifier.h"
#include "lbl/query_cache.h"

// cdec headers
#include "ff.h"
#include "hg.h"
#include "sentence_metadata.h"

#define kORDER 5

using namespace std;
using namespace oxlm;
namespace po = boost::program_options;

void ParseOptions(
    const string& input, string& filename, string& feature_name,
    bool& cache_queries) {
  po::options_description options("LBL language model options");
  options.add_options()
      ("file,f", po::value<string>()->required(),
          "File containing serialized language model")
      ("name,n", po::value<string>()->default_value("LBLLM"),
          "Feature name")
      ("cache-queries", "Whether should cache n-gram probabilities.");

  po::variables_map vm;
  vector<string> args;
  boost::split(args, input, boost::is_any_of(" "));
  po::store(po::command_line_parser(args).options(options).run(), vm);
  po::notify(vm);

  filename = vm["file"].as<string>();
  feature_name = vm["name"].as<string>();
  cache_queries = vm.count("cache-queries");
}

struct SimplePair {
  SimplePair() : first(), second() {}
  SimplePair(double x, double y) : first(x), second(y) {}
  SimplePair& operator+=(const SimplePair& o) { first += o.first; second += o.second; return *this; }
  double first;
  double second;
};

class FF_LBLLM : public FeatureFunction {
 public:
  FF_LBLLM(
      const string& filename, const string& feature_name,
      const bool& cache_queries)
      : fid(FD::Convert(feature_name)),
        fid_oov(FD::Convert(feature_name + "_OOV")),
        processIdentifier("FF_LBLLM"),
        cacheQueries(cache_queries), cacheHits(0), totalHits(0) {
    loadLanguageModel(filename);

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

    cerr << "Initializing map contents (map size=" << dict.max() << ")\n";
    for (int i = 1; i < dict.max(); ++i)
      AddToWordMap(i);
    cerr << "Done.\n";
    ss_off = OrderToStateSize(kORDER)-1;  // offset of "state size" member
    FeatureFunction::SetStateSize(OrderToStateSize(kORDER));
    kSTART = dict.Convert("<s>");
    kSTOP = dict.Convert("</s>");
    kUNKNOWN = dict.Convert("<unk>");
    kNONE = -1;
    kSTAR = dict.Convert("<{STAR}>");
    last_id = 0;
  }

  virtual void PrepareForInput(const SentenceMetadata& smeta) {
    unsigned id = smeta.GetSentenceID();
    if (last_id > id) {
      cerr << "last_id = " << last_id << " but id = " << id << endl;
      abort();
    }
    last_id = id;
    lm->clear_cache();
  }

  inline void AddToWordMap(const WordID lbl_id) {
    const unsigned cdec_id = TD::Convert(dict.Convert(lbl_id));
    assert(cdec_id > 0);
    if (cdec_id >= cdec2lbl.size())
      cdec2lbl.resize(cdec_id + 1);
    cdec2lbl[cdec_id] = lbl_id;
  }

 protected:
  void TraversalFeaturesImpl(const SentenceMetadata&,
                             const HG::Edge& edge,
                             const vector<const void*>& ant_states,
                             SparseVector<double>* features,
                             SparseVector<double>* estimated_features,
                             void* state) const {
    SimplePair ft = LookupWords(*edge.rule_, ant_states, state);
    if (ft.first) features->set_value(fid, ft.first);
    if (ft.second) features->set_value(fid_oov, ft.second);
    ft = EstimateProb(state);
    if (ft.first) estimated_features->set_value(fid, ft.first);
    if (ft.second) estimated_features->set_value(fid_oov, ft.second);
  }

  void FinalTraversalFeatures(const void* ant_state,
                              SparseVector<double>* features) const {
    const SimplePair ft = FinalTraversalCost(ant_state);
    if (ft.first) features->set_value(fid, ft.first);
    if (ft.second) features->set_value(fid_oov, ft.second);
  }

 private:
  void loadLanguageModel(const string& filename) {
    Time start_time = GetTime();
    cerr << "Reading LM from " << filename << "..." << endl;
    ifstream ifile(filename);
    if (!ifile.good()) {
      cerr << "Failed to open " << filename << " for reading" << endl;
      abort();
    }
    boost::archive::binary_iarchive ia(ifile);
    ia >> lm;
    dict = lm->label_set();
    Time stop_time = GetTime();
    cerr << "Reading language model took " << GetDuration(start_time, stop_time)
         << " seconds..." << endl;
  }

  // returns the lbl equivalent of a cdec word or kUNKNOWN
  inline unsigned ConvertCdec(unsigned w) const {
    int res = 0;
    if (w < cdec2lbl.size())
      res = cdec2lbl[w];
    if (res) return res; else return kUNKNOWN;
  }

  inline int StateSize(const void* state) const {
    return *(static_cast<const char*>(state) + ss_off);
  }

  inline void SetStateSize(int size, void* state) const {
    *(static_cast<char*>(state) + ss_off) = size;
  }

  inline double WordProb(WordID word, const WordID* history) const {
    vector<WordID> context;
    for (int i = 0; i < kORDER - 1 && history && (*history != kNONE); ++i) {
      context.push_back(*history++);
    }

    if (!context.empty() && context.back() == kSTART) {
      context.resize(kORDER - 1, kSTART);
    } else {
      context.resize(kORDER - 1, kUNKNOWN);
    }

    double score;

    if (!cacheQueries) {
      score = lm->log_prob(word, context, true, true);
    } else {
      NGramQuery query(word, context);
      ++totalHits;
      pair<double, bool> ret = cache.get(query);
      if (ret.second) {
        ++cacheHits;
        score = ret.first;
      } else {
        score = lm->log_prob(word, context, true, true);
        cache.put(query, score);
      }
    }

    return score;
  }

  // first = prob, second = unk
  inline SimplePair LookupProbForBufferContents(int i) const {
    if (buffer_[i] == kUNKNOWN)
      return SimplePair(0.0, 1.0);
    double p = WordProb(buffer_[i], &buffer_[i+1]);
    return SimplePair(p, 0.0);
  }

  inline SimplePair ProbNoRemnant(int i, int len) const {
    int edge = len;
    bool flag = true;
    SimplePair sum;
    while (i >= 0) {
      if (buffer_[i] == kSTAR) {
        edge = i;
        flag = false;
      } else if (buffer_[i] <= 0) {
        edge = i;
        flag = true;
      } else {
        if ((edge-i >= kORDER) || (flag && !(i == (len-1) && buffer_[i] == kSTART)))
          sum += LookupProbForBufferContents(i);
      }
      --i;
    }
    return sum;
  }

  SimplePair EstimateProb(const vector<WordID>& phrase) const {
    cerr << "EstimateProb(&phrase): ";
    int len = phrase.size();
    buffer_.resize(len + 1);
    buffer_[len] = kNONE;
    int i = len - 1;
    for (int j = 0; j < len; ++j,--i)
      buffer_[i] = phrase[j];
    return ProbNoRemnant(len - 1, len);
  }

  //Vocab_None is (unsigned)-1 in srilm, same as kNONE. in srilm (-1), or that SRILM otherwise interprets -1 as a terminator and not a word
  SimplePair EstimateProb(const void* state) const {
    //cerr << "EstimateProb(*state): ";
    int len = StateSize(state);
    //  << "residual len: " << len << endl;
    buffer_.resize(len + 1);
    buffer_[len] = kNONE;
    const int* astate = reinterpret_cast<const WordID*>(state);
    int i = len - 1;
    for (int j = 0; j < len; ++j,--i) {
      buffer_[i] = astate[j];
    }
    return ProbNoRemnant(len - 1, len);
  }

  // for <s> (n-1 left words) and (n-1 right words) </s>
  SimplePair FinalTraversalCost(const void* state) const {
    //cerr << "FinalTraversalCost(*state): ";
    int slen = StateSize(state);
    int len = slen + 2;
    // cerr << "residual len: " << len << endl;
    buffer_.resize(len + 1);
    buffer_[len] = kNONE;
    buffer_[len-1] = kSTART;
    const int* astate = reinterpret_cast<const WordID*>(state);
    int i = len - 2;
    for (int j = 0; j < slen; ++j,--i)
      buffer_[i] = astate[j];
    buffer_[i] = kSTOP;
    assert(i == 0);
    //cerr << "FINAL: ";
    return ProbNoRemnant(len - 1, len);
  }

  //NOTE: this is where the scoring of words happens (heuristic happens in EstimateProb)
  SimplePair LookupWords(const TRule& rule, const vector<const void*>& ant_states, void* vstate) const {
    //cerr << "LookupWords(*vstate=" << vstate << ")";
    int len = rule.ELength() - rule.Arity();
    for (unsigned i = 0; i < ant_states.size(); ++i)
      len += StateSize(ant_states[i]);
    buffer_.resize(len + 1);
    buffer_[len] = kNONE;
    int i = len - 1;
    const vector<WordID>& e = rule.e();
    for (unsigned j = 0; j < e.size(); ++j) {
      if (e[j] < 1) {
        const int* astate = reinterpret_cast<const int*>(ant_states[-e[j]]);
        int slen = StateSize(astate);
        for (int k = 0; k < slen; ++k)
          buffer_[i--] = astate[k];
      } else {
        buffer_[i--] = ConvertCdec(e[j]);
      }
    }

    SimplePair sum;
    int* remnant = reinterpret_cast<int*>(vstate);
    int j = 0;
    i = len - 1;
    int edge = len;

    while (i >= 0) {
      if (buffer_[i] == kSTAR) {
        edge = i;
      } else if (edge-i >= kORDER) {
        //cerr << "X: ";
        sum += LookupProbForBufferContents(i);
      } else if (edge == len && remnant) {
        remnant[j++] = buffer_[i];
      }
      --i;
    }
    if (!remnant) return sum;

    if (edge != len || len >= kORDER) {
      remnant[j++] = kSTAR;
      if (kORDER-1 < edge) edge = kORDER-1;
      for (int i = edge-1; i >= 0; --i)
        remnant[j++] = buffer_[i];
    }

    SetStateSize(j, vstate);
    return sum;
  }

  static int OrderToStateSize(int order) {
    return order>1 ?
      ((order-1) * 2 + 1) * sizeof(WordID) + 1
      : 0;
  }

  virtual ~FF_LBLLM() {
    if (cacheQueries) {
      cerr << "Cache hit ratio: " << Real(cacheHits) / totalHits << endl;

      ofstream f(cacheFile);
      boost::archive::binary_oarchive ia(f);
      cerr << "Saving n-gram probability cache to " << cacheFile << endl;
      ia << cache;
      cerr << "Finished saving " << cache.size()
           << " n-gram probabilities..." << endl;

      processIdentifier.freeId(processId);
      cerr << "Freed id " << processId
           << " at time " << Clock::to_time_t(GetTime()) << endl;
    }
  }

  oxlm::Dict dict;
  mutable vector<WordID> buffer_;
  int ss_off;
  WordID kSTART;
  WordID kSTOP;
  WordID kUNKNOWN;
  WordID kNONE;
  WordID kSTAR;
  const int fid;
  const int fid_oov;
  vector<int> cdec2lbl;
  unsigned last_id;

  boost::shared_ptr<FactoredNLM> lm;

  ProcessIdentifier processIdentifier;
  int processId;

  bool cacheQueries;
  string cacheFile;
  mutable QueryCache cache;
  mutable int cacheHits, totalHits;
};

extern "C" FeatureFunction* create_ff(const string& str) {
  string filename, feature_name;
  bool cache_queries;
  ParseOptions(str, filename, feature_name, cache_queries);
  return new FF_LBLLM(filename, feature_name, cache_queries);
}
