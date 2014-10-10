#pragma once

#include "lbl/cdec_state_converter_base.h"

namespace oxlm {

class CdecStateConverter : public CdecStateConverterBase {
 public:
  CdecStateConverter(int state_offset);

  vector<int> getTerminals(const void* state) const;

  void convert(void* state, const vector<int>& terminals) const;
};

} // namespace oxlm
