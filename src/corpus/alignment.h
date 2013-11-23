#ifndef _ALIGNMENT_H
#define _ALIGNMENT_H

#include<vector>
#include<boost/lexical_cast.hpp>
#include<boost/bimap/multiset_of.hpp>
#include<boost/bimap.hpp>
#include<iostream>
#include<assert.h>
#include<memory>

typedef boost::bimap< boost::bimaps::multiset_of<int>, 
        boost::bimaps::multiset_of<int>, 
        boost::bimaps::set_of_relation<> > Alignment;
typedef Alignment::relation AlignmentPoint;
typedef std::vector<Alignment> Alignments;
typedef std::shared_ptr<Alignment> AlignmentPtr;

inline bool read_alignment(std::istream &in, AlignmentPtr a)
{
  std::string buf, token;
  if (!std::getline(in, buf)) return false;

  std::istringstream ss(buf);
  while(ss >> token) {
    size_t hyphen = token.find('-');
    assert(hyphen != std::string::npos);
    int s=boost::lexical_cast<int,std::string>(token.substr(0,hyphen));
    int t=boost::lexical_cast<int,std::string>(token.substr(hyphen+1,token.size()-hyphen-1));
    a->insert(AlignmentPoint(s,t));
  }

  return true;
}

inline std::ostream& print_alignment(std::ostream& out, const Alignment& alignment)
{
  if (alignment.empty())
    return out;

  Alignment::const_iterator it=alignment.begin();
  out << it->left << "-" << it->right;
  ++it;
  for (; it != alignment.end(); ++it)
    out << " " << it->left << "-" << it->right;

  return out;
}

struct AlignmentEvaluation {
  AlignmentEvaluation(float p, float r) : precision(p), recall(r), f_score(2.0*p*r/(p+r)) {};
  float precision;
  float recall;
  float f_score;
};

inline std::ostream& 
operator<<(std::ostream& out, const AlignmentEvaluation& ae) {
  out << "Alignment Evaluation: Recall=" << ae.recall << " Precision=" << ae.precision << " F-Score=" << ae.f_score;
  return out;
}

inline AlignmentEvaluation evaluate_alignment(const Alignments& references, const Alignments& predictions)
{
  int reference_points=0, predicted=0, correct=0;
  int num_alignments = std::min(predictions.size(), references.size());

  for (int i=0; i < num_alignments; ++i) {
    for (Alignment::const_iterator alignment_it=references.at(i).begin(); alignment_it != references.at(i).end(); ++alignment_it) {
      Alignment::const_iterator find_result = predictions.at(i).find(*alignment_it);
      if (find_result != predictions.at(i).end()) correct++;
    }
    reference_points += references.at(i).size();
    predicted += predictions.at(i).size();
    assert(reference_points >= correct);
    assert(predicted >= correct);
  }

  return AlignmentEvaluation((float) correct/predicted, (float) correct/reference_points);
}

#endif // _ALIGNMENT_H
