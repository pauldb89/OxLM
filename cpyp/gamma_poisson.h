#ifndef _CPYP_GAMMA_POISSON_H_
#define _CPYP_GAMMA_POISSON_H_

#include "m.h"

namespace cpyp {

// http://en.wikipedia.org/wiki/Conjugate_prior
template <typename F>
struct gamma_poisson {
  gamma_poisson(F shape, F rate) :
    a(shape), b(rate), n(), marginal(), llh() {}

  F prob(unsigned x) const {
    return exp(M<F>::log_negative_binom(x, a + marginal, 1.0 - (b + n) / (1 + b + n)));
  }

  void increment(unsigned x) {
    llh += M<F>::log_negative_binom(x, a + marginal, 1.0 - (b + n) / (1 + b + n));
    ++n;
    marginal += x;
  }

  void decrement(unsigned x) {
    --n;
    marginal -= x;
    llh -= M<F>::log_negative_binom(x, a + marginal, 1.0 - (b + n) / (1 + b + n));
  }

  F log_likelihood() const {
    return llh;
  }

  // if you want to infer (a,b), you'll need a likelihood function that
  // takes (a',b') as parameters and evaluates the likelihood. working out this
  // likelihood will mean you need to keep a list of the observations. since i
  // don't think in general you'd bother to infer (a,b), i didn't implement
  // this since it comes at the cost of more memory usage
  F a, b;   // \lambda ~ Gamma(a,b)

  unsigned n, marginal;
  double llh;
};

}

#endif
