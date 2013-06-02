#ifndef _TIED_RESAMPLER_H_
#define _TIED_RESAMPLER_H_

#include <set>
#include <vector>
#include "random.h"
#include "slice_sampler.h"
#include "m.h"

namespace oxlm {

// tie together CRPs that are conditionally independent given their hyperparameters
template <class CRP>
struct tied_parameter_resampler {
  explicit tied_parameter_resampler(double da, double db, double ss, double sr, double d=0.5, double s=1.0) :
      d_alpha(da),
      d_beta(db),
      s_shape(ss),
      s_rate(sr),
      discount(d),
      strength(s) {}

  void insert(CRP* crp) {
    crps.insert(crp);
    crp->set_discount(discount);
    crp->set_strength(strength);
    assert(!crp->has_discount_prior());
    assert(!crp->has_strength_prior());
  }

  void erase(CRP* crp) {
    crps.erase(crp);
  }

  size_t size() const {
    return crps.size();
  }

  double log_likelihood(double d, double s) const {
    if (s <= -d) return -std::numeric_limits<double>::infinity();
    double llh = Md::log_beta_density(d, d_alpha, d_beta) +
                 Md::log_gamma_density(d + s, s_shape, s_rate);
    for (auto& crp : crps) { llh += crp->log_likelihood(d, s); }
    return llh;
  }

  double log_likelihood() const {
    return log_likelihood(discount, strength);
  }

  template<typename Engine>
  void resample_hyperparameters(Engine& eng, const unsigned nloop = 5, const unsigned niterations = 10) {
    if (size() == 0) { std::cerr << "EMPTY - not resampling\n"; return; }
    for (unsigned iter = 0; iter < nloop; ++iter) {
      strength = slice_sampler1d([this](double prop_s) { return this->log_likelihood(discount, prop_s); },
                              strength, eng, -discount + std::numeric_limits<double>::min(),
                              std::numeric_limits<double>::infinity(), 0.0, niterations, 100*niterations);
      double min_discount = std::numeric_limits<double>::min();
      if (strength < 0.0) min_discount -= strength;
      discount = slice_sampler1d([this](double prop_d) { return this->log_likelihood(prop_d, strength); },
                          discount, eng, min_discount,
                          1.0, 0.0, niterations, 100*niterations);
    }
    strength = slice_sampler1d([this](double prop_s) { return this->log_likelihood(discount, prop_s); },
                            strength, eng, -discount + std::numeric_limits<double>::min(),
                            std::numeric_limits<double>::infinity(), 0.0, niterations, 100*niterations);
    std::cerr << "Resampled " << crps.size() << " CRPs (d=" << discount << ",s="
              << strength << ") = " << log_likelihood(discount, strength) << std::endl;
    for (auto& crp : crps)
      crp->set_hyperparameters(discount, strength);
  }

  double get_discount() const { return discount; }
  double get_strength() const { return strength; }

 private:
  std::set<CRP*> crps;
  const double d_alpha, d_beta, s_shape, s_rate;
  double discount, strength;
};

// split according to some criterion
template <class CRP>
struct bintied_parameter_resampler {
  explicit bintied_parameter_resampler(unsigned nbins) :
      resamplers(nbins, tied_parameter_resampler<CRP>(1,1,1,1)) {}

  void insert(unsigned bin, CRP* crp) {
    resamplers[bin].insert(crp);
  }

  void erase(unsigned bin, CRP* crp) {
    resamplers[bin].erase(crp);
  }

  template <typename Engine>
  void resample_hyperparameters(Engine& eng) {
    for (unsigned i = 0; i < resamplers.size(); ++i) {
      std::cerr << "BIN " << i << " (" << resamplers[i].size() << " CRPs): " << std::flush;
      resamplers[i].resample_hyperparameters(eng);
    }
  }

  double log_likelihood() const {
    double llh = 0;
    for (unsigned i = 0; i < resamplers.size(); ++i)
      llh += resamplers[i].log_likelihood();
    return llh;
  }

 private:
  std::vector<tied_parameter_resampler<CRP> > resamplers;
};

}

#endif
