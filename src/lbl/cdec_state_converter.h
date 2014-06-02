#pragma once

#include <vector>

using namespace std;

namespace oxlm {

class CdecStateConverter {
 public:
  CdecStateConverter(int state_offset);

  vector<int> convert(const void* state) const;

  void convert(const vector<int>& symbols, void* state) const;

 private:
  int getStateSize(const void* state) const;

  void setStateSize(void* state, int state_size) const;

  int stateOffset;
};

} // namespace oxlm
