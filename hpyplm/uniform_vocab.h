#ifndef _UNIFORM_VOCAB_H_
#define _UNIFORM_VOCAB_H_

#include <cassert>
#include <vector>

namespace oxlm {

// uniform distribution over a fixed vocabulary
struct UniformVocabulary {
  UniformVocabulary(size_t vs, double, double, double, double) : p0(1.0 / vs), draws() {}
  template<typename Engine>
  void increment(int, const std::vector<int>&, Engine&) { ++draws; }
  template<typename Engine>
  void decrement(int, const std::vector<int>&, Engine&) { --draws; assert(draws >= 0); }
  double prob(int, const std::vector<int>&) const { return p0; }
  template<typename Engine>
  void resample_hyperparameters(Engine&) {}
  double log_likelihood() const { return draws * log(p0); }
  template<class Archive> void serialize(Archive& ar, const unsigned int) {
    ar & p0;
    ar & draws;
  }
  double p0;
  int draws;
};

}

#endif
