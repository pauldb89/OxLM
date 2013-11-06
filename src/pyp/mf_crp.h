#ifndef _CPYP_MF_CRP_H_
#define _CPYP_MF_CRP_H_

#include <iostream>
#include <numeric>
#include <cassert>
#include <cmath>
#include <utility>
#include <unordered_map>
#include <functional>
#include "random.h"
#include "slice_sampler.h"
#include "crp_table_manager.h"
#include "m.h"

namespace pyp {

// Chinese restaurant process (Pitman-Yor parameters) histogram-based table tracking
// based on the implementation proposed by Blunsom et al. 2009
//
// this implementation assumes that the observation likelihoods are either 1 (if they
// are identical to the "parameter" drawn from G_0) or 0. This is fine for most NLP
// applications but violated in PYP mixture models etc.
template <unsigned NumFloors, typename Dish, typename DishHash = std::hash<Dish> >
class mf_crp {
 public:
  mf_crp() :
      num_tables_(),
      num_customers_(),
      discount_(0.1),
      strength_(1.0),
      discount_prior_strength_(std::numeric_limits<double>::quiet_NaN()),
      discount_prior_beta_(std::numeric_limits<double>::quiet_NaN()),
      strength_prior_shape_(std::numeric_limits<double>::quiet_NaN()),
      strength_prior_rate_(std::numeric_limits<double>::quiet_NaN()) {
    check_hyperparameters();
  }

  mf_crp(double disc, double strength) :
      num_tables_(),
      num_customers_(),
      discount_(disc),
      strength_(strength),
      discount_prior_strength_(std::numeric_limits<double>::quiet_NaN()),
      discount_prior_beta_(std::numeric_limits<double>::quiet_NaN()),
      strength_prior_shape_(std::numeric_limits<double>::quiet_NaN()),
      strength_prior_rate_(std::numeric_limits<double>::quiet_NaN()) {
    check_hyperparameters();
  }

  mf_crp(double d_strength, double d_beta, double c_shape, double c_rate, double d = 0.8, double c = 1.0) :
      num_tables_(),
      num_customers_(),
      discount_(d),
      strength_(c),
      discount_prior_strength_(d_strength),
      discount_prior_beta_(d_beta),
      strength_prior_shape_(c_shape),
      strength_prior_rate_(c_rate) {
    check_hyperparameters();
  }

  void check_hyperparameters() {
    if (discount_ < 0.0 || discount_ >= 1.0) {
      std::cerr << "Bad discount: " << discount_ << std::endl;
      abort();
    }
    if (strength_ <= -discount_) {
      std::cerr << "Bad strength: " << strength_ << " (discount=" << discount_ << ")" << std::endl;
      abort();
    }

    llh_ = lgamma(strength_) - lgamma(strength_ / discount_);
    if (has_discount_prior())
      llh_ = Md::log_beta_density(discount_, discount_prior_strength_, discount_prior_beta_);
    if (has_strength_prior())
      llh_ += Md::log_gamma_density(strength_ + discount_, strength_prior_shape_, strength_prior_rate_);
    if (num_tables_ > 0) llh_ = log_likelihood(discount_, strength_);
  }

  double discount() const { return discount_; }
  double strength() const { return strength_; }
  void set_hyperparameters(double d, double s) {
    discount_ = d; strength_ = s;
    check_hyperparameters();
  }
  void set_discount(double d) { discount_ = d; check_hyperparameters(); }
  void set_strength(double a) { strength_ = a; check_hyperparameters(); }

  bool has_discount_prior() const {
    return !std::isnan(discount_prior_strength_);
  }

  bool has_strength_prior() const {
    return !std::isnan(strength_prior_shape_);
  }

  void clear() {
    num_tables_ = 0;
    num_customers_ = 0;
    dish_locs_.clear();
  }

  unsigned num_tables() const {
    return num_tables_;
  }

  unsigned num_tables(const Dish& dish) const {
    auto it = dish_locs_.find(dish);
    if (it == dish_locs_.end()) return 0;
    return it->second.num_tables();
  }

  unsigned num_customers() const {
    return num_customers_;
  }

  unsigned num_customers(const Dish& dish) const {
    auto it = dish_locs_.find(dish);
    if (it == dish_locs_.end()) return 0;
    return it->num_customers();
  }

  // returns (floor,table delta) where table delta +1 or 0 indicates whether a new table was opened or not
  template <class InputIterator, class InputIterator2, typename Engine>
  std::pair<unsigned,int> increment(const Dish& dish, InputIterator p0i, InputIterator2 lambdas, Engine& eng) {
    typedef decltype(*p0i + 0.0) F;

    const F marginal_p0 = std::inner_product(p0i, p0i + NumFloors, lambdas, F(0.0));
    if (marginal_p0 > F(1.000001)) {
      std::cerr << "bad marginal: " << marginal_p0 << std::endl;
      abort();
    }

    crp_table_manager<NumFloors>& loc = dish_locs_[dish];
    bool share_table = false;
    if (loc.num_customers()) {
      const F p_empty = F(strength_ + num_tables_ * discount_) * marginal_p0;
      const F p_share = F(loc.num_customers() - loc.num_tables() * discount_);
      share_table = sample_bernoulli(p_empty, p_share, eng);
    }

    unsigned floor = 0;
    if (share_table) {
      unsigned n = loc.share_table(discount_, eng);
      update_llh_add_customer_to_table_seating(n);
    } else {
      if (NumFloors > 1) { // sample floor
        F r = F(sample_uniform01<double>(eng)) * marginal_p0;
        for (unsigned i = 0; i < NumFloors; ++i) {
          r -= (*p0i) * (*lambdas);
          ++p0i;
          ++lambdas;
          if (r <= F(0.0)) { floor = i; break; }
        }
      }
      loc.create_table(floor);
      update_llh_add_customer_to_table_seating(0);
      ++num_tables_;
    }
    ++num_customers_;
    return std::make_pair(floor, share_table ? 0 : 1);
  }

  // returns -1 or 0, indicating whether a table was closed
  // logq = probability that the selected table will be reselected if
  //     increment_no_base is called with dish [optional]
  template<typename Engine>
  std::pair<unsigned,int> decrement(const Dish& dish, Engine& eng, double* logq = nullptr) {
    crp_table_manager<NumFloors>& loc = dish_locs_[dish];
    assert(loc.num_customers());
    if (loc.num_customers() == 1) {
      update_llh_remove_customer_from_table_seating(1);
      unsigned floor = 0;
      for (; floor < NumFloors; ++floor)
        if (!loc.h[floor].empty()) break;
      assert(floor < NumFloors);
      
      dish_locs_.erase(dish);
      --num_tables_;
      --num_customers_;
      // q = 1 since this is the first customer
      return std::make_pair(floor, -1);
    } else {
      unsigned selected_table_postcount = 0;
      const std::pair<unsigned,int> delta = loc.remove_customer(eng, &selected_table_postcount);
      update_llh_remove_customer_from_table_seating(selected_table_postcount + 1);
      --num_customers_;
      if (delta.second) --num_tables_;

      if (logq) {
        double p_empty = (strength_ + num_tables_ * discount_);
        double p_share = (loc.num_customers() - loc.num_tables() * discount_);
        const double z = p_empty + p_share;
        p_empty /= z;
        p_share /= z;
        if (selected_table_postcount)
          *logq += log(p_share * (selected_table_postcount - discount_) /
                     (loc.num_customers() - loc.num_tables() * discount_));
        else
          *logq += log(p_empty);
      }
      return delta;
    }
  }

  template <class InputIterator, class InputIterator2>
  decltype(**((InputIterator*) 0) + 0.0) prob(const Dish& dish, InputIterator p0i, InputIterator2 lambdas) const {
    typedef decltype(*p0i + 0.0) F;
    const F marginal_p0 = std::inner_product(p0i, p0i + NumFloors, lambdas, F(0.0));
    if (marginal_p0 >= F(1.000001)) {
      std::cerr << "bad marginal: " << marginal_p0 << std::endl;
      abort();
    }
    if (num_tables_ == 0) return marginal_p0;

    auto it = dish_locs_.find(dish);
    const F r = F(num_tables_ * discount_ + strength_);
    if (it == dish_locs_.end()) {
      return r * marginal_p0 / F(num_customers_ + strength_);
    } else {
      return (F(it->second.num_customers() - discount_ * it->second.num_tables()) + r * marginal_p0) /
                   F(num_customers_ + strength_);
    }
  }

  double log_likelihood() const {
    return llh_;
//    return log_likelihood(discount_, strength_);
  }

  // call this before changing the number of tables / customers
  void update_llh_add_customer_to_table_seating(unsigned n) {
    unsigned t = 0;
    if (n == 0) t = 1;
    llh_ -= log(strength_ + num_customers_);
    if (t == 1) llh_ += log(discount_) + log(strength_ / discount_ + num_tables_);
    if (n > 0) llh_ += log(n - discount_);
  }

  // call this before changing the number of tables / customers
  void update_llh_remove_customer_from_table_seating(unsigned n) {
    unsigned t = 0;
    if (n == 1) t = 1;
    llh_ += log(strength_ + num_customers_ - 1);
    if (t == 1) llh_ -= log(discount_) + log(strength_ / discount_ + num_tables_ - 1);
    if (n > 1) llh_ -= log(n - discount_ - 1);
  }

  // adapted from http://en.wikipedia.org/wiki/Chinese_restaurant_process
  // does not include P_0's
  double log_likelihood(const double& discount, const double& strength) const {
    double lp = 0.0;
    if (has_discount_prior())
      lp = Md::log_beta_density(discount, discount_prior_strength_, discount_prior_beta_);
    if (has_strength_prior())
      lp += Md::log_gamma_density(strength + discount, strength_prior_shape_, strength_prior_rate_);
    assert(lp <= 0.0);
    if (num_customers_) {  // if restaurant is not empty
      if (discount > 0.0) {  // two parameter case: discount > 0
        const double r = lgamma(1.0 - discount);
        if (strength)
          lp += lgamma(strength) - lgamma(strength / discount);
        lp += - lgamma(strength + num_customers_)
             + num_tables_ * log(discount) + lgamma(strength / discount + num_tables_);
        // above line implies
        // 1) when adding a customer to a restaurant containing N customers:
        //    lp -= log(strength + N)    [because \Gamma(s+N+1) = (s+N)\Gamma(s+N)
        // 2) when removing a customer from a restaurant containing N customers:
        //    lp += log(strength + N - 1)  [because \Gamma(s+N) = (s+N-1)\Gamma(s+N-1)]
        // 3) when adding a table to a restaurant containing T tables:
        //    lp += log(discount) + log(s / d + T)
        // 4) when removing a table from a restuarant containint T tables:
        //    lp -= log(discount) + log(s / d + T - 1)

        assert(std::isfinite(lp));
        for (auto& dish_loc : dish_locs_)
          for (unsigned floor = 0; floor < NumFloors; ++floor)
            for (auto& bin : dish_loc.second.h[floor])
              lp += (lgamma(bin.first - discount) - r) * bin.second;
         // above implies
         // 1) when adding to a table seating N > 1 customers
         //    lp += log(N - discount)
         // 2) when adding a new table
         //    do nothing
         // 3) when removing a customer from a table with N > 1 customers
         //    lp -= log(N - discount - 1)
         // 4) when closing a table
         //    do nothing
      } else if (!discount) { // discount == 0.0 (ie, Dirichlet Process)
        lp += lgamma(strength) + num_tables_ * log(strength) - lgamma(strength + num_tables_);
        assert(std::isfinite(lp));
        for (auto& dish_loc : dish_locs_)
          lp += lgamma(dish_loc.second.num_tables());
      } else { // should never happen
        assert(!"discount less than 0 detected!");
      }
    }
    assert(std::isfinite(lp));
    return lp;
  }

  template<typename Engine>
  void resample_hyperparameters(Engine& eng, const unsigned nloop = 5, const unsigned niterations = 10) {
    assert(has_discount_prior() || has_strength_prior());
    if (num_customers() == 0) return;
    double s = strength_;
    double d = discount_;
    for (unsigned iter = 0; iter < nloop; ++iter) {
      if (has_strength_prior()) {
        s = slice_sampler1d([this,d](double prop_s) { return this->log_likelihood(d, prop_s); },
                            s, eng, -d + std::numeric_limits<double>::min(),
                            std::numeric_limits<double>::infinity(), 0.0, niterations, 100*niterations);
      }
      if (has_discount_prior()) {
        double min_discount = std::numeric_limits<double>::min();
        if (s < 0.0) min_discount -= s;
        d = slice_sampler1d([this,s](double prop_d) { return this->log_likelihood(prop_d, s); },
                            d, eng, min_discount,
                            1.0, 0.0, niterations, 100*niterations);
      }
    }
    s = slice_sampler1d([this,d](double prop_s) { return this->log_likelihood(d, prop_s); },
                        s, eng, -d + std::numeric_limits<double>::min(),
                        std::numeric_limits<double>::infinity(), 0.0, niterations, 100*niterations);
    set_hyperparameters(d, s);
  }

  void print(std::ostream* out) const {
    std::cerr << "PYP(d=" << discount_ << ",c=" << strength_ << ") customers=" << num_customers_ << std::endl;
    for (auto& dish_loc : dish_locs_)
      (*out) << dish_loc.first << " : " << dish_loc.second << std::endl;
  }

  typedef typename std::unordered_map<Dish, crp_table_manager<NumFloors>, DishHash>::const_iterator const_iterator;
  const_iterator begin() const {
    return dish_locs_.begin();
  }
  const_iterator end() const {
    return dish_locs_.end();
  }

  void swap(crp<Dish>& b) {
    std::swap(num_tables_, b.num_tables_);
    std::swap(num_customers_, b.num_customers_);
    std::swap(dish_locs_, b.dish_locs_);
    std::swap(discount_, b.discount_);
    std::swap(strength_, b.strength_);
    std::swap(discount_prior_strength_, b.discount_prior_strength_);
    std::swap(discount_prior_beta_, b.discount_prior_beta_);
    std::swap(strength_prior_shape_, b.strength_prior_shape_);
    std::swap(strength_prior_rate_, b.strength_prior_rate_);
    std::swap(llh_, b.llh_);
  }

  template<class Archive> void serialize(Archive& ar, const unsigned int version) {
    ar & num_tables_;
    ar & num_customers_;
    ar & discount_;
    ar & strength_;
    ar & discount_prior_strength_;
    ar & discount_prior_beta_;
    ar & strength_prior_shape_;
    ar & strength_prior_rate_;
    ar & llh_;  // llh of current partition structure
    ar & dish_locs_;
  }
 private:
  unsigned num_tables_;
  unsigned num_customers_;
  std::unordered_map<Dish, crp_table_manager<NumFloors>, DishHash> dish_locs_;

  double discount_;
  double strength_;

  // optional beta prior on discount_ (NaN if no prior)
  double discount_prior_strength_;
  double discount_prior_beta_;

  // optional gamma prior on strength_ (NaN if no prior)
  double strength_prior_shape_;
  double strength_prior_rate_;

  double llh_;  // llh of current partition structure
};

template<unsigned N,typename T>
void swap(mf_crp<N,T>& a, mf_crp<N,T>& b) {
  a.swap(b);
}

template <unsigned N,typename T,typename H>
std::ostream& operator<<(std::ostream& o, const mf_crp<N,T,H>& c) {
  c.print(&o);
  return o;
}

}

#endif
