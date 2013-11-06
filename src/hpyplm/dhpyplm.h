#ifndef _DHPYPLM_H_
#define _DHPYPLM_H_

#include <unordered_map>
#include <vector>

#include "corpus/corpus.h"
#include "pyp/m.h"
#include "pyp/random.h"
#include "pyp/crp.h"
#include "pyp/mf_crp.h"
#include "pyp/tied_parameter_resampler.h"

#include "hpyplm/uvector.h"
#include "hpyplm/uniform_vocab.h"
#include "hpyplm/hpyplm.h"

namespace oxlm {

// A not very memory-efficient implementation of a domain adapting
// HPYP language model, as described by Wood & Teh (AISTATS, 2009)
//

// represents an N-gram domain adapted LM
template <unsigned N> struct DAPYPLM;

// zero-gram model
template<> struct DAPYPLM<0> : PYPLM<0> {
  DAPYPLM(PYPLM<0>& rllm) : PYPLM(rllm) {}
};

template <unsigned N> struct DAPYPLM {
  DAPYPLM(PYPLM<N>& rllm) : path(1,1,1,1,0.1,1.0), tr(1,1,1,1), in_domain_backoff(rllm.backoff), llm(rllm), lookup(N-1) {}
  template<typename Engine>
  void increment(unsigned w, const std::vector<unsigned>& context, Engine& eng) {
    const double p0[2]{in_domain_backoff.prob(w, context), llm.prob(w, context)};
    double b = path.prob(0, 0.5);
    const double lam[2]{b, 1.0 - b};
    for (unsigned i = 0; i < N-1; ++i)
      lookup[i] = context[context.size() - 1 - i];
    auto it = p.find(lookup);
    if (it == p.end()) {
      it = p.insert(std::make_pair(lookup, pyp::mf_crp<2, unsigned>(0.8,1))).first;
      tr.insert(&it->second);  // add to resampler
    }
    const std::pair<unsigned, int> floor_count = it->second.increment(w, p0, lam, eng);
    if (floor_count.second) {
      if (floor_count.first == 0) { // in-domain backoff
        //cerr << "Increment<" << N << "> in domain\n";
        path.increment(0, 0.5, eng);
        in_domain_backoff.increment(w, context, eng);
      } else { // domain general backoff
        //cerr << "Increment<" << N << "> out of domain\n";
        path.increment(1, 0.5, eng);
        llm.increment(w, context, eng);
      }
    }
  }

  template<typename Engine>
  void decrement(unsigned w, const std::vector<unsigned>& context, Engine& eng) {
    for (unsigned i = 0; i < N-1; ++i)
      lookup[i] = context[context.size() - 1 - i];
    auto it = p.find(lookup);
    assert(it != p.end());
    const std::pair<unsigned, int> floor_count = it->second.decrement(w, eng);
    //cerr << "Dec: floor=" << floor_count.first << endl;
    if (floor_count.second) {
      if (floor_count.first == 0) { // in-domain backoff
        //cerr << "Decrement<" << N << "> in domain\n";
        path.decrement(0, eng);
        in_domain_backoff.decrement(w, context, eng);
      } else { // domain general backoff
        //cerr << "Decrement<" << N << "> out of domain\n";
        path.decrement(1, eng);
        llm.decrement(w, context, eng);
      }
    }
  }

  double prob(unsigned w, const std::vector<unsigned>& context) const {
    const double p0[2]{in_domain_backoff.prob(w, context), llm.prob(w, context)};
    double b = path.prob(0, 0.5);
    const double lam[2]{b, 1.0 - b};
    for (unsigned i = 0; i < N-1; ++i)
      lookup[i] = context[context.size() - 1 - i];
    auto it = p.find(lookup);
    if (it == p.end()) return lam[0] * p0[0] + lam[1] * p0[1];
    return it->second.prob(w, p0, lam);
  }

  double log_likelihood() const {
    return path.log_likelihood() + path.num_customers() * log(0.5) + tr.log_likelihood();
  }

  template<typename Engine>
  void resample_hyperparameters(Engine& eng) {
    path.resample_hyperparameters(eng);
    std::cerr << "Path<" << N << "> d=" << path.discount() << ",s=" << path.strength() << " p(in_domain) = " << path.prob(0, 0.5) << std::endl;
    tr.resample_hyperparameters(eng);
    in_domain_backoff.resample_hyperparameters(eng);
  }

  template<class Archive> void serialize(Archive& ar, const unsigned int version) {
    ar & path;
    in_domain_backoff.serialize(ar, version);
    ar & p;
  }

  pyp::crp<unsigned> path;
  pyp::tied_parameter_resampler<pyp::mf_crp<2, unsigned>> tr;
  DAPYPLM<N-1> in_domain_backoff;
  PYPLM<N>& llm;
  mutable std::vector<unsigned> lookup;  // thread-local
  std::unordered_map<std::vector<unsigned>, pyp::mf_crp<2, unsigned>, uvector_hash> p;  // .first = context .second = 2-floor CRP
};

}

#endif
