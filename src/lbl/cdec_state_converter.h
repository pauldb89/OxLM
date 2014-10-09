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

class CdecStateConverter : public CdecStateConverterBase {
public:
  CdecStateConverter(int state_offset);
  vector<int> getTerminals(const void* state) const;
  void convert(void* state, const vector<int>& terminals) const;
};

class CdecConditionalStateConverter : public CdecStateConverterBase {
 public:
  CdecConditionalStateConverter(int state_offset);
  vector<int> getTerminals(const void* state) const;
  vector<int> getAffiliations(const void* state) const;

  int getLeftmostLinkSource(const void* state) const;
  int getLeftmostLinkDistance(const void* state) const;
  int getRightmostLinkSource(const void* state) const;
  int getRightmostLinkDistance(const void* state) const;
  int getTargetSpanSize(const void* state) const;
  int getSourceSpanStart(const void* state) const;
  int getSourceSpanEnd(const void* state) const;

  void convert(void* state,
    const vector<int>& terminals, const vector<int>& affiliations,
    int leftMostLinkSource, int leftMostLinkDistance,
    int rightMostLinkSource, int rightMostLinkDistance,
    int sourceSpanStart, int sourceSpanEnd, int targetSpanSize) const;
private:
  // In the source conditional LM, a state has a bunch of extra fields
  // since we have to keep track of which source words each target
  // terminal is affiliated with.
  static const int extraConditionalInfoSize = 7;
};

} // namespace oxlm
