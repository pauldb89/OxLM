#ifndef HCHPYPLM_H_
#define HCHPYPLM_H_

#include <vector>
#include <unordered_map>

#include "hpyplm.h"
#include "pyp/m.h"
#include "corpus/corpus.h"
#include "pyp/random.h"

#include "hpyplm/uvector.h"
#include "hpyplm/uniform_vocab.h"

// An implementation of an (NW,NC)-gram LM based on PYPs that predicts each character 
// of a word conditioned on the previous NW words and NC characters.

namespace oxlm {

// represents an (NW,NC)-gram LM which predicts each character in a word
template <unsigned NW, unsigned NC> struct CHPYPLM {
  explicit CHPYPLM(const Dict& d, double da = 1.0, double db = 1.0, double ss = 1.0, double sr = 1.0) 
    : pyplm(d.max(), da, db, ss, sr), lookup_(NW+NC-2), dict_(d), 
      sos_index_(dict_.Lookup(sos_)), space_index_(dict_.Lookup(space_))  {}

  template<typename Engine>
  void increment(const std::string& w, const std::vector<unsigned>& context, Engine& eng) {
    copy_word_context(context, lookup_);
    for (size_t i=0; i<=w.size(); ++i) {
      copy_character_context(w, i, lookup_);
      pyplm.increment(i == w.size() ? space_index_ : dict_.Lookup(w.substr(i,1)), lookup_, eng);
    }
  }

  template<typename Engine>
  void decrement(const std::string& w, const std::vector<unsigned>& context, Engine& eng) {
    copy_word_context(context, lookup_);
    for (size_t i=0; i<=w.size(); ++i) {
      copy_character_context(w, i, lookup_);
      pyplm.decrement(i == w.size() ? space_index_ : dict_.Lookup(w.substr(i,1)), lookup_, eng);
    }
  }

  double prob(const std::string& w, const std::vector<unsigned>& context) const {
    double p=1.0;
    copy_word_context(context, lookup_);
    for (size_t i=0; i<=w.size(); ++i) {
      copy_character_context(w, i, lookup_);
      p *= pyplm.prob(i == w.size() ? space_index_ : dict_.Lookup(w.substr(i,1)), lookup_);
    }
    return p;
  }

  template<typename Engine>
  std::string generate(const std::vector<unsigned>& context, Engine& eng) {
    std::string result("");
    std::vector<double> probs(dict_.max()+1,0);
    copy_word_context(context, lookup_);
    while (true) {
      copy_character_context(result, result.size(), lookup_);
      for (unsigned c=0; c<=dict_.max(); ++c)
        probs.at(c) = pyplm.prob(c, lookup_);

      pyp::multinomial_distribution<double> dist(probs);
      unsigned index = dist(eng);
      //pyplm.increment(index, lookup_, eng);

      if (index == space_index_) break;
      result.append(dict_.Convert(index));
    }
    return result;
  }

  double log_likelihood() const 
  { return pyplm.log_likelihood(); }

  template<typename Engine>
  void resample_hyperparameters(Engine& eng) 
  { pyplm.resample_hyperparameters(eng); }

  template<class Archive> void serialize(Archive& ar, const unsigned int version) { 
    ar & pyplm; 
    ar & sos_index_; 
    ar & space_index_; 
    ar & dict_; 
  }


  PYPLM<NW+NC-1> pyplm;

  std::ostream& print(std::ostream& out) const {
    out << "\n(" << NW << "," << NC << ")-gram CHPYPLM:\n";
    pyplm.print(out);
    return out;
  }

private:
  void copy_character_context(const std::string& w, int w_i, std::vector<unsigned>& result) const {
    assert(w_i <= static_cast<int>(w.size()));
    int r=0;
    for (int i=w_i-NC+1; i < w_i; ++i,++r) {
      assert(NW-1+i >= 0);
      assert(NW-1+r < NW+NC-2);
      result.at(NW-1+r) = (i < 0 ? sos_index_: dict_.Lookup(w.substr(i,1)));
    }
  }

  void copy_word_context(const std::vector<unsigned>& context, std::vector<unsigned>& result) const {
    assert (context.size() >= NW-1);
    for (unsigned i = 0; i < NW-1; ++i) {
      assert(NW-2-i >= 0);
      result.at(NW-2-i) = context.at(context.size()-1-i);
    }
  }

  mutable std::vector<unsigned> lookup_;
  const Dict& dict_;
  const std::string sos_ = "<s>";
  const std::string space_ = " ";
  unsigned sos_index_;
  unsigned space_index_;
};

}

#endif
