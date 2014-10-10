#include "gtest/gtest.h"

#include "lbl/cdec_conditional_state_converter.h"

namespace oxlm {

TEST(CdecConditionalStateConverterTest, TestConvertFromState) {
  CdecConditionalStateConverter converter(52);

  char* state = new char[53];
  int* new_state = reinterpret_cast<int*>(state);

  // symbols
  new_state[0] = 5;
  new_state[1] = 10;
  new_state[2] = 7;
  // alignments
  new_state[3] = 0;
  new_state[4] = -1;
  new_state[5] = 2;
  // span size
  new_state[6] = 3;
  // left-most link
  new_state[7] = 0;
  // left-most distance
  new_state[8] = 0;
  // right-most link
  new_state[9] = 2;
  // right-most distance
  new_state[10] = 0;
  // source span start
  new_state[11] = 0;
  // source span end
  new_state[12] = 2;

  state[52] = 13;

  vector<int> expected_symbols = {5, 10, 7};
  EXPECT_EQ(expected_symbols, converter.getTerminals(state));
  vector<int> expected_affiliations = {0, -1, 2};
  EXPECT_EQ(expected_affiliations, converter.getAffiliations(state));
  EXPECT_EQ(3, converter.getTargetSpanSize(state));
  EXPECT_EQ(0, converter.getLeftmostLinkSource(state));
  EXPECT_EQ(0, converter.getLeftmostLinkDistance(state));
  EXPECT_EQ(2, converter.getRightmostLinkSource(state));
  EXPECT_EQ(0, converter.getRightmostLinkDistance(state));
  EXPECT_EQ(0, converter.getSourceSpanStart(state));
  EXPECT_EQ(2, converter.getSourceSpanEnd(state));

  delete state;
}

TEST(CdecConditionalStateConverterTest, TestConvertToState) {
  char* state = new char[53];
  int* new_state = reinterpret_cast<int*>(state);

  CdecConditionalStateConverter converter(52);
  vector<int> symbols = {5, 10, 7};
  vector<int> affiliations = {0, -1, 2};

  converter.convert(state, symbols, affiliations, 3, 0, 0, 2, 0, 0, 2);

  for (size_t i = 0; i < symbols.size(); ++i) {
    EXPECT_EQ(symbols[i], new_state[i]);
  }

  for (size_t i = 0; i < affiliations.size(); ++i) {
    EXPECT_EQ(affiliations[i], new_state[3 + i]);
  }

  EXPECT_EQ(3, new_state[6]);
  EXPECT_EQ(0, new_state[7]);
  EXPECT_EQ(0, new_state[8]);
  EXPECT_EQ(2, new_state[9]);
  EXPECT_EQ(0, new_state[10]);
  EXPECT_EQ(0, new_state[11]);
  EXPECT_EQ(2, new_state[12]);

  delete state;
}

} // namespace oxlm
