#pragma once

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "lbl/vocabulary.h"

using namespace std;

namespace oxlm {

class CdecLBLMapper {
 public:
  CdecLBLMapper(const boost::shared_ptr<Vocabulary>& vocab);

  int convert(int cdec_id) const;

 private:
  void add(int lbl_id, int cdec_id);

  boost::shared_ptr<Vocabulary> vocab;
  vector<int> cdec2lbl;
  int kUNKNOWN;
};

} // namespace oxlm
