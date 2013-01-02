#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <iostream>
#include <fstream>
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/archive/text_oarchive.hpp>

struct UnigramDistribution {
  std::map<double, std::string> prob_to_token;
  std::map<std::string, double> token_to_prob;

  void read(const std::string& filename) {
    std::ifstream file(filename.c_str());
    std::cerr << "Reading unigram distribution from " 
      << filename.c_str() << "." << std::endl;

    double sum=0;
    std::string key, value;
    while (file >> value >> key) {
      double v = boost::lexical_cast<double>(value);
      sum += v;
      prob_to_token.insert(std::make_pair(sum, key));   
      token_to_prob.insert(std::make_pair(key, v));   
    }
  }

  double prob(const std::string& s) const {
   std::map<std::string, double>::const_iterator it
     = token_to_prob.find(s);
   return it != token_to_prob.end() ? it->second : 0.0; 
 }

  bool empty() const { return prob_to_token.empty(); }
};

struct ModelData {
  enum ZType { Sampled, Exact };

  ModelData() : ztype(Sampled), step_size(0.1), eta_t0(1), l2_parameter(0.0), 
                time_series_parameter(1.0), label_sample_size(100), 
                feature_type("explicit"), hash_bits(16), threads(1), 
                iteration_size(1), verbose(false), ngram_order(3), 
                word_representation_size(100), uniform(false)
  {}

  ZType       ztype;
  float       step_size;
  float       eta_t0;
  float       l2_parameter;
  float       time_series_parameter;
  int         label_sample_size;
  std::string feature_type;
  int         hash_bits;
  int         threads;
  int         iteration_size;
  bool        verbose;
  int         ngram_order;
  int         word_representation_size;
  bool        uniform;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & ztype;
    ar & step_size;
    ar & eta_t0;
    ar & l2_parameter;
    ar & time_series_parameter;
    ar & label_sample_size;
    ar & feature_type;
    ar & hash_bits;
    ar & threads;
    ar & iteration_size;
    ar & verbose;
    ar & ngram_order;
    ar & word_representation_size;
    ar & uniform;
  }
};
typedef boost::shared_ptr<ModelData> ModelDataPtr;

#endif // _CONFIG_H_
