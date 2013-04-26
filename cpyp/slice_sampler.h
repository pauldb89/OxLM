//! slice-sampler.h is an MCMC slice sampler
//!
//! Mark Johnson, 1st August 2008
//! A few changes by Chris Dyer, Aug 13, 2012

#ifndef SLICE_SAMPLER_H
#define SLICE_SAMPLER_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

namespace cpyp {

//! slice_sampler_rfc_type{} returns the value of a user-specified
//! function if the argument is within range, or - infinity otherwise
//
template <typename F, typename Fn, typename U>
struct slice_sampler_rfc_type {
  F min_x, max_x;
  const Fn& f;
  U max_nfeval, nfeval;
  slice_sampler_rfc_type(F min_x, F max_x, const Fn& f, U max_nfeval) 
    : min_x(min_x), max_x(max_x), f(f), max_nfeval(max_nfeval), nfeval(0) { }
    
  F operator() (F x) {
    if (min_x < x && x < max_x) {
      assert(++nfeval <= max_nfeval);
      F fx = f(x);
      assert(std::isfinite(fx));
      return fx;
    }
      return -std::numeric_limits<F>::infinity();
  }
};  // slice_sampler_rfc_type{}

//! slice_sampler1d() implements the univariate "range doubling" slice sampler
//! described in Neal (2003) "Slice Sampling", The Annals of Statistics 31(3), 705-767.
//
template <typename F, typename LogF, typename Engine>
F slice_sampler1d(const LogF& logF0,               //!< log of function to sample
		  F x,                             //!< starting point
		  Engine& eng,                     //!< random number engine
		  F min_x = -std::numeric_limits<F>::infinity(),  //!< minimum value of support
		  F max_x = std::numeric_limits<F>::infinity(),   //!< maximum value of support
		  F w = 0.0,                       //!< guess at initial width
		  unsigned nsamples=1,             //!< number of samples to draw
		  unsigned max_nfeval=200)         //!< max number of function evaluations
{
  typedef unsigned U;

  std::uniform_real_distribution<F> u01(F(0), F(1)); 
  slice_sampler_rfc_type<F,LogF,U> logF(min_x, max_x, logF0, max_nfeval);

  assert(std::isfinite(x));

  if (w <= 0.0) {                           // set w to a default width 
    if (min_x > -std::numeric_limits<F>::infinity() && max_x < std::numeric_limits<F>::infinity())
      w = (max_x - min_x)/4;
    else
      w = std::max(((x < 0.0) ? -x : x)/4, (F) 0.1);
  }
  assert(std::isfinite(w));

  F logFx = logF(x);
  for (U sample = 0; sample < nsamples; ++sample) {
    F logY = logFx + log(u01(eng)+1e-100);     //! slice logFx at this value
    assert(std::isfinite(logY));

    F xl = x - w*u01(eng);                     //! lower bound on slice interval
    F logFxl = logF(xl);
    F xr = xl + w;                          //! upper bound on slice interval
    F logFxr = logF(xr);

    while (logY < logFxl || logY < logFxr)  // doubling procedure
      if (u01(eng) < 0.5) 
	logFxl = logF(xl -= xr - xl);
      else
	logFxr = logF(xr += xr - xl);
	
    F xl1 = xl;
    F xr1 = xr;
    while (true) {                          // shrinking procedure
      F x1 = xl1 + u01(eng)*(xr1 - xl1);
      if (logY < logF(x1)) {
	F xl2 = xl;                         // acceptance procedure
	F xr2 = xr; 
	bool d = false;
	while (xr2 - xl2 > 1.1*w) {
	  F xm = (xl2 + xr2)/2;
	  if ((x < xm && x1 >= xm) || (x >= xm && x1 < xm))
	    d = true;
	  if (x1 < xm)
	    xr2 = xm;
	  else
	    xl2 = xm;
	  if (d && logY >= logF(xl2) && logY >= logF(xr2))
	    goto unacceptable;
	}
	x = x1;
	goto acceptable;
      }
      goto acceptable;
    unacceptable:
      if (x1 < x)                           // rest of shrinking procedure
	xl1 = x1;
      else 
	xr1 = x1;
    }
  acceptable:
    w = (4*w + (xr1 - xl1))/5;              // update width estimate
  }
  return x;
}

}

#endif  // SLICE_SAMPLER_H
