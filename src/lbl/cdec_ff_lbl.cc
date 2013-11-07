#include <iostream>
#include <string>

#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

// pyp stuff
#include "corpus/corpus.h"
#include "lbl/nlm.h"

// cdec stuff
#include "stringlib.h"
#include "ff.h"
#include "hg.h"
#include "fdict.h"
#include "sentence_metadata.h"

#define kORDER 5

using namespace std;

namespace {

bool parse_lmspec(std::string const& in, string &featurename, string &filename, string& reffile) {
  vector<string> const& argv=SplitOnWhitespace(in);
  featurename="LBLLM";
#define LMSPEC_NEXTARG if (i==argv.end()) {            \
    cerr << "Missing argument for "<<*last<<". "; goto usage; \
    } else { ++i; }

  for (vector<string>::const_iterator last,i=argv.begin(),e=argv.end();i!=e;++i) {
    string const& s=*i;
    if (s[0]=='-') {
      if (s.size()>2) goto fail;
      switch (s[1]) {
      case 'r':
        LMSPEC_NEXTARG; reffile=*i;
        break;
      case 'n':
        LMSPEC_NEXTARG; featurename=*i;
        break;
#undef LMSPEC_NEXTARG
      default:
      fail:
        cerr<<"Unknown LanguageModel option "<<s<<" ; ";
        goto usage;
      }
    } else {
      if (filename.empty())
        filename=s;
      else {
        cerr<<"More than one filename provided. ";
        goto usage;
      }
    }
  }
  if (!filename.empty())
    return true;
usage:
  cerr<<" LBLLM parse error needs filename and optional -n FeatureName\n";
  return false;
}

} // namespace

struct SimplePair {
  SimplePair() : first(), second() {}
  SimplePair(double x, double y) : first(x), second(y) {}
  SimplePair& operator+=(const SimplePair& o) { first += o.first; second += o.second; return *this; }
  double first;
  double second;
};

class FF_LBLLM : public FeatureFunction {
 public:
  FF_LBLLM(const string& lm_file, const string& feat, const string& reffile)
    : lm(ModelData(), oxlm::Dict(), true), fid(FD::Convert(feat)), fid_oov(FD::Convert(feat+"_OOV"))
    //: lm(ModelData(), oxlm::Dict(), true, std::vector<int>()), fid(FD::Convert(feat)), fid_oov(FD::Convert(feat+"_OOV"))
  {
    {
      cerr << "Reading LM from " << lm_file << " ...\n";
      //ifstream ifile(lm_file.c_str(), ios::in | ios::binary);
      ifstream ifile(lm_file.c_str(), ios::in);
      if (!ifile.good()) {
        cerr << "Failed to open " << lm_file << " for reading\n";
        abort();
      }
      boost::archive::text_iarchive ia(ifile);
      ia >> lm;
      dict = lm.label_set();
    }
    /*
    {
      ifstream z_ifile((lm_file+".z").c_str(), ios::in);
      if (!z_ifile.good()) {
        cerr << "Failed to open " << (lm_file+".z") << " for reading\n";
        abort();
      }
      cerr << "Reading LM Z from " << lm_file+".z" << " ...\n";
      boost::archive::text_iarchive ia(z_ifile);
      ia >> z_approx;
    }
    */

    cerr << "Initializing map contents (map size=" << dict.max() << ")\n";
    for (int i = 1; i < dict.max(); ++i)
      AddToWordMap(i);
    cerr << "Done.\n";
    ss_off = OrderToStateSize(kORDER)-1;  // offset of "state size" member
    FeatureFunction::SetStateSize(OrderToStateSize(kORDER));
    kSTART = dict.Convert("<s>");
    kSTOP = dict.Convert("</s>");
    kUNKNOWN = dict.Convert("_UNK_");
    kNONE = -1;
    kSTAR = dict.Convert("<{STAR}>");
    last_id = 0;

    // optional online "adaptation" by training on previous references
    if (reffile.size()) {
      cerr << "Reference file: " << reffile << endl;
      set<WordID> rv;
      oxlm::ReadFromFile(reffile, &dict, &ref_sents, &rv);
    }
  }

  virtual void PrepareForInput(const SentenceMetadata& smeta) {
    unsigned id = smeta.GetSentenceID();
    if (last_id > id) {
      cerr << "last_id = " << last_id << " but id = " << id << endl;
      abort();
    }
    /*
    if (last_id < ref_sents.size() && last_id < id) {
      cerr << "  ** LBLLM: Adapting LM using previously translated references for sentences [" << last_id << ',' << id << ")\n";
      for (unsigned i = last_id; i < id; ++i)
        if (i < ref_sents.size()) IncorporateSentenceToLM(ref_sents[i]);
    }
    */
    last_id = id;
//    cerr << "\n  Cached contexts: " << lm.m_context_cache.size() << endl;
    lm.clear_cache();
  }

  inline void AddToWordMap(const WordID lbl_id) {
    const unsigned cdec_id = TD::Convert(dict.Convert(lbl_id));
    assert(cdec_id > 0);
    if (cdec_id >= cdec2lbl.size())
      cdec2lbl.resize(cdec_id + 1);
    cdec2lbl[cdec_id] = lbl_id;
  }
/*
  void IncorporateSentenceToLM(const vector<WordID>& sent) {
    oxlm::MT19937 eng;
    vector<WordID> ctx(kORDER - 1, kSTART);
    for (auto w : sent) {
      AddToWordMap(w);
      lm.increment(w, ctx, eng);
      ctx.push_back(w);
    }
  }
*/
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

  inline double WordProb(WordID word, WordID const* context) const {
    vector<WordID> xx;
    for (int i=0; i<kORDER-1 && context && (*context != kNONE); ++i) {
      xx.push_back(*context++);
    }
    //if (xx.size() != kORDER-1) return 0;

    if (!xx.empty() && xx.back() == kSTART)
      xx.resize(kORDER-1, kSTART);
    else
      xx.resize(kORDER-1, kUNKNOWN);

    reverse(xx.begin(),xx.end());
//    cerr << "LEN=" << xx.size() << ", (";
//    for (unsigned j = 0; j < xx.size(); ++j)
//      cerr << dict.Convert(xx[j]) << " ";
//    cerr << " | " << dict.Convert(word);

    double s = lm.log_prob(word, xx, true, false);

//    cerr << "), s = " << s << endl;

    return s;
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
    for (int j = 0; j < len; ++j,--i)
      buffer_[i] = astate[j];
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

  oxlm::Dict dict;
  mutable vector<WordID> buffer_;
  int ss_off;
  WordID kSTART;
  WordID kSTOP;
  WordID kUNKNOWN;
  WordID kNONE;
  WordID kSTAR;
  //oxlm::PYPLM<kORDER> lm;
  //oxlm::NLM lm;
  //oxlm::NLMApproximateZ z_approx;
  oxlm::FactoredOutputNLM lm;
  const int fid;
  const int fid_oov;
  vector<int> cdec2lbl; // cdec2lbl[TD::Convert("word")] returns the index in the lbl model

  // stuff for online updating of LM
  vector<vector<WordID>> ref_sents;
  unsigned last_id; // id of the last sentence that was translated
};

extern "C" FeatureFunction* create_ff(const string& str) {
  string featurename, filename, reffile;
  if (!parse_lmspec(str, featurename, filename, reffile))
    abort();
  return new FF_LBLLM(filename, featurename, reffile);
}


