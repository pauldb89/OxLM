#pragma once

#include <boost/shared_ptr.hpp>

#include "lbl/word_to_class_index.h"

namespace oxlm {

class ClassHashSpaceDecider {
 public:
  ClassHashSpaceDecider(
      const boost::shared_ptr<WordToClassIndex>& index, int hash_space);

  int getHashSpace(int class_id) const;

 private:
  vector<int> classHashSpaces;
};

} // namespace oxlm
