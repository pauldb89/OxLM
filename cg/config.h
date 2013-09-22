#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <iostream>
#include <fstream>
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/archive/text_oarchive.hpp>

namespace oxlm {

struct ModelData {
  enum ZType { Sampled, Exact };

  ModelData() : step_size(0.1), l2_parameter(0.0), source_l2_parameter(0.0), threads(1), 
                iteration_size(1), verbose(false), ngram_order(3), word_representation_size(100), 
                classes(1), nonlinear(false), diagonal(false), source_window_width(-1)
  {}

  float       step_size;
  float       l2_parameter;
  float       source_l2_parameter;
  int         threads;
  int         iteration_size;
  bool        verbose;
  int         ngram_order;
  int         word_representation_size;
  int         classes;
  bool        nonlinear;
  bool        diagonal;
  int         source_window_width;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & step_size;
    ar & l2_parameter;
    ar & source_l2_parameter;
    ar & threads;
    ar & iteration_size;
    ar & verbose;
    ar & ngram_order;
    ar & word_representation_size;
    ar & classes;
    ar & nonlinear;
    ar & diagonal;
    ar & source_window_width;
  }
};
typedef boost::shared_ptr<ModelData> ModelDataPtr;

} // namespace oxlm
#endif // _CONFIG_H_
