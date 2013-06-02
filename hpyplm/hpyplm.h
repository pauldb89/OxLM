#ifndef HPYPLM_H_
#define HPYPLM_H_

#include <vector>
#include <unordered_map>

#include "pyp/m.h"
#include "pyp/random.h"
#include "pyp/crp.h"
#include "pyp/tied_parameter_resampler.h"

#include "hpyplm/uvector.h"
#include "hpyplm/uniform_vocab.h"

#include "corpus/corpus.h"

// An implementation of an N-gram LM based on PYPs as described 
// in Y.-W. Teh. (2006) A Hierarchical Bayesian Language Model
// based on Pitman-Yor Processes. In Proc. ACL.

namespace oxlm {

template <unsigned N> struct PYPLM;

template<> struct PYPLM<0> : public UniformVocabulary {
  PYPLM(unsigned vs, double a, double b, double c, double d) 
    : UniformVocabulary(vs, a, b, c, d) {}

  std::ostream& print(std::ostream& out) const 
  { out << "Uniform base CRP\n"; return out; }
};

// represents an N-gram LM
template <unsigned N> struct PYPLM {
  PYPLM() : backoff(0,1,1,1,1), tr(1,1,1,1,0.8,0.0), lookup(N-1) {
    oxlm::MT19937 eng;
    tr.insert(&singleton_crp);  // add to resampler
    singleton_crp.increment(0, 1.0, eng);
  }

  explicit PYPLM(unsigned vs, double da = 1.0, double db = 1.0, double ss = 1.0, double sr = 1.0) 
    : backoff(vs, da, db, ss, sr), tr(da, db, ss, sr, 0.8, 0.0), lookup(N-1) {
      oxlm::MT19937 eng;
      tr.insert(&singleton_crp);  // add to resampler
      singleton_crp.increment(0, 1.0, eng);
    }

  template<typename Engine>
  void increment(WordId w, const std::vector<WordId>& context, Engine& eng) {
    const double bo = backoff.prob(w, context);
    copy_context(context, lookup);
    auto it = p.find(lookup);
    if (it == p.end()) {
      
      // note the first customer of a crp but don't create a crp object
      auto s_result = singletons.insert(make_pair(lookup, w));
      if (s_result.second) {
        backoff.increment(w, context, eng);
        return;
      }

      // second customer triggers crp creation
      it = p.insert(make_pair(lookup, oxlm::crp<unsigned>(0.8,0))).first;
      tr.insert(&it->second);  // add to resampler
      
      // insert the first customer, which by definition creates a table
      assert (it->second.num_tables() == 0);
      it->second.increment(s_result.first->second, 1.0, eng);
      singletons.erase(s_result.first);
    }
    if (it->second.increment(w, bo, eng))
      backoff.increment(w, context, eng);
  }

  template<typename Engine>
  void decrement(WordId w, const std::vector<WordId>& context, Engine& eng) {
    copy_context(context, lookup);

    // check if this is a customer of a singleton context
    auto s_it = singletons.find(lookup);
    if (s_it != singletons.end()) {
      assert (s_it->second == w);
      backoff.decrement(w, context, eng);
      singletons.erase(s_it);
      return;
    }

    // don't delete restaurants that become singleton as they will probably 
    // cease to be so shortly
    auto it = p.find(lookup);
    assert(it != p.end());
    if (it->second.decrement(w, eng))
      backoff.decrement(w, context, eng);
  }

  double prob(WordId w, const std::vector<WordId>& context) const {
    const double bo = backoff.prob(w, context);
    copy_context(context, lookup);

    // singletons can be scored with a default crp
    auto s_it = singletons.find(lookup);
    if (s_it != singletons.end())
      // All singleton customers eat the dish 0, so if w matches the singleton 
      // for this context score 0, otherwise score 1 which is not in the crp.
      return singleton_crp.prob(s_it->second == w ? 0 : 1, bo);

    auto it = p.find(lookup);
    if (it == p.end()) return bo;
    return it->second.prob(w, bo);
  }

  double log_likelihood() const {
    return backoff.log_likelihood() + tr.log_likelihood() + singleton_log_likelihood();
  }

  template<typename Engine>
  void resample_hyperparameters(Engine& eng) {
    tr.resample_hyperparameters(eng);
    backoff.resample_hyperparameters(eng);
  }

  template<class Archive> void serialize(Archive& ar, const unsigned int version) {
    backoff.serialize(ar, version);
    ar & p;
    ar & singletons;
    ar & singleton_crp;
  }

  PYPLM<N-1> backoff;
  oxlm::tied_parameter_resampler<oxlm::crp<unsigned>> tr;
  mutable std::vector<WordId> lookup;  // thread-local
  std::unordered_map<std::vector<WordId>, oxlm::crp<unsigned>, vector_hash> p;  // .first = context .second = CRP
  std::unordered_map<std::vector<WordId>, WordId, vector_hash> singletons; // contexts seen exactly once

  std::ostream& print(std::ostream& out) const {
    out << '\n' << N << "-gram PYLM:\n";
    if (!singletons.empty()) {
      out << "Singleton CRP: " << singleton_crp;
      for (auto r : singletons) {
        for(auto w : r.first) 
          out << w << ' ';
        out << r.second << '\n';
      }
    }
    out << "Non-singleton CRPs: \n";
    for (auto r : p) {
      for(auto w : r.first) out << w << ' ';
      r.second.print(&out);
    }
    backoff.print(out);
    return out;
  }

private:
  void copy_context(const std::vector<WordId>& context, std::vector<WordId>& result) const {
    assert (context.size() >= N-1);
    for (unsigned i = 0; i < N-1; ++i)
      result[i] = context[context.size() - 1 - i];
  }

  double singleton_log_likelihood() const {
    return singletons.size()*singleton_crp.log_likelihood(tr.get_discount(), tr.get_strength());
  }

  oxlm::crp<unsigned> singleton_crp;
};

}

#endif
