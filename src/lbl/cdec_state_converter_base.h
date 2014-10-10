#pragma once

#include <vector>

using namespace std;

namespace oxlm {

class CdecStateConverterBase {
 public:
  CdecStateConverterBase(int state_offset);

  virtual vector<int> getTerminals(const void* state) const = 0;

  int getStateSize(const void* state) const;

  void setStateSize(void* state, int state_size) const;

 private:
  int stateOffset;
};

} // namespace oxlm
