#ifndef _UTILS_H_
#define _UTILS_H_

// STL
#include <vector>
#include <map>
#include <limits>

// Eigen
#include <Eigen/Dense>

namespace oxlm {

typedef float Real;
typedef std::vector<Real> Reals;
typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> MatrixReal;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1>              VectorReal;
typedef Eigen::Array<Real, Eigen::Dynamic, 1>               ArrayReal;
typedef boost::shared_ptr<MatrixReal>                       MatrixRealPtr;
typedef boost::shared_ptr<VectorReal>                       VectorRealPtr;

typedef std::vector<WordId> Sentence;
typedef std::vector<WordId> Corpus;

typedef int TrainingInstance;
typedef std::vector<TrainingInstance> TrainingInstances;


inline VectorReal softMax(const VectorReal& v) {
  Real max = v.maxCoeff();
  return (v.array() - (log((v.array() - max).exp().sum()) + max)).exp();
}

inline VectorReal logSoftMax(const VectorReal& v, Real* lz=0) {
  Real max = v.maxCoeff();
  Real log_z = log((v.array() - max).exp().sum()) + max;
  if (lz!=0) *lz = log_z;
  return v.array() - log_z;
}


template <typename T>
struct Log
{
    static T zero() { return -std::numeric_limits<T>::infinity(); } 

    static T add(T l1, T l2)
    {
        if (l1 == zero()) return l2;
        if (l1 > l2) 
            return l1 + std::log(1 + exp(l2 - l1));
        else
            return l2 + std::log(1 + exp(l1 - l2));
    }

    static T subtract(T l1, T l2)
    {
        //std::assert(l1 >= l2);
        return l1 + log(1 - exp(l2 - l1));
    }
};


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


} // namespace oxlm
#endif // _UTILS_H_
