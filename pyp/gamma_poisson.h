#ifndef _CPYP_GAMMA_POISSON_H_
#define _CPYP_GAMMA_POISSON_H_

#include "m.h"

namespace pyp {

// http://en.wikipedia.org/wiki/Conjugate_prior
template <typename F>
struct gamma_poisson {
  gamma_poisson(F shape, F rate) :
    a(shape), b(rate), n(), marginal() {}

  F prob(unsigned x) const {
    return exp(M<F>::log_negative_binom(x, a + marginal, 1.0 - (b + n) / (1 + b + n)));
  }

  void increment(unsigned x) {
    ++n;
    marginal += x;
  }

  void decrement(unsigned x) {
    --n;
    marginal -= x;
  }

  F log_likelihood() const {
    // TODO
    return 0;
  }

  F a, b;
  unsigned n, marginal;
};

}

#endif
