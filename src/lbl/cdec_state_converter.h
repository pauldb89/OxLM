#pragma once

#include <vector>

using namespace std;

namespace oxlm {

class CdecStateConverter {
 public:
  CdecStateConverter(int state_offset);

  vector<int> getTerminals(const void* state, bool hasSpan = false) const;
  vector<int> getAffiliations(const void* state) const;

  int getLeftmostLinkSource(const void* state) const;
  int getLeftmostLinkDistance(const void* state) const;
  int getRightmostLinkSource(const void* state) const;
  int getRightmostLinkDistance(const void* state) const;
  int getTargetSpanSize(const void* state) const;
  int getSourceSpanStart(const void* state) const;
  int getSourceSpanEnd(const void* state) const;

  void convert(void* state, const vector<int>& terminals) const;
  void convert(void* state,
    const vector<int>& terminals, const vector<int>& affiliations,
    int leftMostLinkSource = -1, int leftMostLinkDistance = -1,
    int rightMostLinkSource = -1, int rightMostLinkDistance = -1,
    int sourceSpanStart = -1, int sourceSpanEnd = -1, int targetSpanSize = -1) const;

// private:
  int getStateSize(const void* state) const;

  void setStateSize(void* state, int state_size) const;
private:
  int stateOffset;
};

} // namespace oxlm
