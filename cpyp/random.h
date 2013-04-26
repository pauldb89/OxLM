#ifndef _CPYP_RANDOM_H_
#define _CPYP_RANDOM_H_

#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <random>

namespace cpyp {

// this is just like std::mt19937 except it initializes itself from /dev/urandom
struct MT19937 : public std::mt19937 {
  MT19937() {
    seed(GetTrulyRandomSeed());
  }
  explicit MT19937(uint32_t s) {
    seed(s);
  }
  static uint32_t GetTrulyRandomSeed() {
    uint32_t seed;
    std::ifstream r("/dev/urandom");
    if (r) {
      r.read((char*)&seed,sizeof(uint32_t));
    }
    if (r.fail() || !r) {
      std::cerr << "Warning: could not read from /dev/urandom. Seeding from clock" << std::endl;
      seed = std::time(NULL);
    }
//    std::cerr << "Seeding random number sequence to " << seed << std::endl;
    return seed;
  }
};

template<typename F, typename Engine>
inline F sample_uniform01(Engine& eng) {
  return std::uniform_real_distribution<F>(0,1)(eng);
}

template<typename F, typename Engine>
inline unsigned sample_bernoulli(const F a, const F b, Engine& eng) {
  const F z = a + b;
  return static_cast<unsigned>(sample_uniform01<F>(eng) > (a / z));
}

// multinomial distribution parameterized by unnormalized probabilities
// F is the type of the probabilities
//   MT19937 eng;
//   vector<float> foo;
//   foo.push_back(1.0);
//   foo.push_back(2.0);
//   foo.push_back(2.0);
//   multinomial_distribution<float> mult(foo);
//   vector<float> hist(3);
//   for (int i = 0; i < 10000; ++i)
//     hist[mult(eng)]++;
template <typename F>
struct multinomial_distribution {
  multinomial_distribution(const std::vector<F>& v) : probs(v), sum(std::accumulate(probs.begin(), probs.end(), F(0))) {}
  template <class Engine>
  unsigned operator()(Engine& eng) const {
    assert(!probs.empty());
    if (probs.size() == 1) return 0;
    const F random = sum * F(sample_uniform01<double>(eng));    // random number between [0 and sum)

    unsigned position = 1;
    F t = probs.at(0);
    for (; position < probs.size() && t < random; ++position)
      t += probs.at(position);
    return position - 1;
  }
  const std::vector<F>& probs;
  const F sum;
};

}

#endif
