#include <Eigen/Dense>

#include "utils/constants.h"

using namespace std;

namespace oxlm {

template<class Scalar>
struct CwiseSetValueOp {
  CwiseSetValueOp(const Scalar& value) : value(value) {}

  const Scalar operator()(const Scalar& x) const {
    return value;
  }

  Scalar value;
};

template<class Scalar>
struct CwiseDenominatorOp {
  CwiseDenominatorOp(Scalar eps) : eps(eps) {}

  const Scalar operator()(const Scalar& x) const {
    return fabs(x) < eps ? 1.0 : 1.0 / x;
  }

  Scalar eps;
};

template<class Scalar>
struct CwiseAdagradUpdateOp {
  CwiseAdagradUpdateOp(const Scalar& step_size) :
      step_size(step_size), eps(EPS * EPS) {}

  Scalar operator()(const Scalar& gradient, const Scalar& adagrad) const {
    return fabs(adagrad) <= eps ? 0 : (step_size * gradient / sqrt(adagrad));
  }

  Scalar step_size, eps;
};

} // namespace oxlm
