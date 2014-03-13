#pragma once

#include <vector>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

using namespace std;

namespace oxlm {

class WordToClassIndex {
 public:
  WordToClassIndex();

  WordToClassIndex(const vector<int>& class_markers);

  int getNumClasses() const;

  int getClass(int word_id) const;

  int getClassMarker(int class_id) const;

  int getClassSize(int class_id) const;

  int getWordIndexInClass(int word_id) const;

  bool operator==(const WordToClassIndex& index) const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & classMarkers;
    ar & wordToClass;
  }

  vector<int> classMarkers;
  vector<int> wordToClass;
};

} // namespace oxlm
