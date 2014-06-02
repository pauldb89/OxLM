#pragma once

#include <vector>

#include <boost/shared_ptr.hpp>

#include "lbl/cdec_lbl_mapper.h"
#include "lbl/cdec_state_converter.h"

using namespace std;

namespace oxlm {

class CdecRuleConverter {
 public:
  CdecRuleConverter(
      const boost::shared_ptr<CdecLBLMapper>& mapper,
      const boost::shared_ptr<CdecStateConverter>& state_converter);

  vector<int> convertTargetSide(
      const vector<int>& target, const vector<const void*>& prev_states) const;

 private:
  boost::shared_ptr<CdecLBLMapper> mapper;
  boost::shared_ptr<CdecStateConverter> stateConverter;
};

} // namespace oxlm
