#ifndef HPYPLM_H_
#define HPYPLM_H_

#include <vector>
#include <unordered_map>

#include "pyp/m.h"
#include "pyp/random.h"
#include "pyp/crp.h"
#include "pyp/tied_parameter_resampler.h"
#include "lbl/log_add.h"

#include "hpyplm/uvector.h"
#include "hpyplm/uniform_vocab.h"

#include "lbl/log_bilinear_model.h"

// An implementation of ...

namespace oxlm {

template <unsigned N> struct LBL_PYPLM;

template<> struct LBL_PYPLM<0> : public UniformVocabulary {
  LBL_PYPLM(const LogBiLinearModel& m, double a, double b, double c, double d,
            MatrixRealPtr pc, VectorRealPtr zc)
    : UniformVocabulary(m.labels(), a, b, c, d), probs_cache(pc), z_cache(zc), 
      p0_array(ArrayReal::Ones(m.labels())*p0) {
//      (*z_cache)(0) = m.B.exp().sum();
      double log_z=Log<double>::zero();
      for (WordId r=0; r < m.labels(); ++r)
        log_z = Log<double>::add(log_z, m.B(r));
      for (WordId r=0; r < m.labels(); ++r) {
        (*probs_cache)(0,r) = exp(m.B(r) - log_z);
        assert(!std::isnan((*probs_cache)(0,r)));
      }
    }

  std::ostream& print(std::ostream& out) const 
  { out << "Uniform base CRP\n"; return out; }

  void recursive_probs(const std::vector<WordId>& context) const {}

  MatrixRealPtr probs_cache;
  VectorRealPtr z_cache;
  ArrayReal p0_array;
};

// represents an N-gram LM
template <unsigned N> struct LBL_PYPLM {
//  LBL_PYPLM() : backoff(0,1,1,1,1), tr(1,1,1,1,0.8,0.0), lookup(N-1), singleton_crp(0.8,0) {
//    pyp::MT19937 eng;
//    tr.insert(&singleton_crp);  // add to resampler
//    singleton_crp.increment(0, 1.0, eng);
//  }

  explicit LBL_PYPLM(const LogBiLinearModel& m, 
                     double da = 1.0, double db = 1.0, double ss = 1.0, double sr = 1.0) 
    : probs_cache(new MatrixReal(N+1,m.labels())), z_cache(new VectorReal(N+1)),
      backoff(m, da, db, ss, sr, probs_cache, z_cache), tr(da, db, ss, sr, 0.8, 0.0), 
      lookup(N-1), singleton_crp(0.8,0), lbl_model(m) {
      assert(lbl_model.config.ngram_order >= (int)N);
      pyp::MT19937 eng;
      tr.insert(&singleton_crp);  // add to resampler
      singleton_crp.increment(0, 1.0, eng);
//      context_products = lbl_model.Q * lbl_model.C.at(N-1);
    }

  explicit LBL_PYPLM(const LogBiLinearModel& m, 
                     double da, double db, double ss, double sr,
                     MatrixRealPtr pc, VectorRealPtr zc)
    : probs_cache(pc), z_cache(zc), backoff(m, da, db, ss, sr, pc, zc), tr(da, db, ss, sr, 0.8, 0.0), 
      lookup(N-1), singleton_crp(0.8,0), lbl_model(m) {
      assert(lbl_model.config.ngram_order >= (int)N);
      pyp::MT19937 eng;
      tr.insert(&singleton_crp);  // add to resampler
      singleton_crp.increment(0, 1.0, eng);

      context_products = lbl_model.Q * lbl_model.C.at(N-1);
    }

  template<typename Engine>
  void increment(WordId w, const std::vector<WordId>& context, Engine& eng) {
    const double bo = backoff_prob(w, context);
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
      it = p.insert(make_pair(lookup, pyp::crp<WordId>(0.8,0))).first;
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
    const double bo = backoff_prob(w, context);
    return prob(w, context, bo);
  }

  double prob(WordId w, const std::vector<WordId>& context, double bo) const {
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

  double backoff_prob(WordId w, const std::vector<WordId>& context) const {
    backoff.recursive_probs(context);
    return (*probs_cache)(N-1,w);
  }

  void recursive_probs(const std::vector<WordId>& context) const {
    backoff.recursive_probs(context);
    double log_z=Log<double>::zero();
    assert (context.size() >= N);
    WordId trigger=context.at(context.size()-N);
    const VectorReal& trigger_vector=context_products.row(trigger);

//    #pragma omp parallel for reduction(+:z)
    for (WordId r=0; r < lbl_model.labels(); ++r) {
      double lbl_weight = lbl_model.R.row(r) * trigger_vector + log(prob(r, context, (*probs_cache)(N-1,r)));
      (*probs_cache)(N,r) = lbl_weight;
      assert(!std::isnan((*probs_cache)(N,r)));
      log_z = Log<double>::add(log_z, lbl_weight);
    }
    (*probs_cache).row(N).array() -= log_z;
    (*probs_cache).row(N) = (*probs_cache).row(N).array().exp();
    for (WordId r=0; r < lbl_model.labels(); ++r)
      assert(!std::isnan((*probs_cache)(N,r)));
//    for (WordId r=0; r < lbl_model.labels(); ++r) {
//      (*probs_cache)(N,r) = lbl_weight * prob(r, context, (*probs_cache)(N-1,r));

    (*z_cache)(N) = exp(log_z);
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

  std::ostream& print(std::ostream& out) const {
    out << '\n' << N << "-gram LBL_PYLM:\n";
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
  MatrixRealPtr probs_cache;
  VectorRealPtr z_cache;
  MatrixReal    context_products;

public:
  LBL_PYPLM<N-1> backoff;
  pyp::tied_parameter_resampler<pyp::crp<WordId>> tr;
  mutable std::vector<WordId> lookup;  // thread-local
  std::unordered_map<std::vector<WordId>, pyp::crp<WordId>, vector_hash> p;  // .first = context .second = CRP
  std::unordered_map<std::vector<WordId>, WordId, vector_hash> singletons; // contexts seen exactly once

private:
  void copy_context(const std::vector<WordId>& context, std::vector<WordId>& result) const {
    assert (context.size() >= N-1);
    for (unsigned i = 0; i < N-1; ++i)
      result[i] = context[context.size() - 1 - i];
  }

  double singleton_log_likelihood() const {
    return singletons.size()*singleton_crp.log_likelihood(tr.get_discount(), tr.get_strength());
  }

  pyp::crp<WordId> singleton_crp;
  const oxlm::LogBiLinearModel& lbl_model;
};

}

#endif
