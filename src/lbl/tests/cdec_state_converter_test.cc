#include "gtest/gtest.h"

#include "lbl/cdec_state_converter.h"

namespace oxlm {

TEST(CdecStateConverterTest, TestConvertFromState) {
  CdecStateConverter converter(16);

  char* state = new char[17];
  int* new_state = reinterpret_cast<int*>(state);

  new_state[0] = 5;
  new_state[1] = 10;
  new_state[2] = 7;
  state[16] = 3;

  vector<int> expected_symbols = {5, 10, 7};
  EXPECT_EQ(expected_symbols, converter.convert(state));

  new_state[3] = 19;
  state[16] = 4;
  expected_symbols = {5, 10, 7, 19};
  EXPECT_EQ(expected_symbols, converter.convert(state));

  delete state;
}

TEST(CdecStateConverterTest, TestConvertToState) {
  CdecStateConverter converter(16);
  char* state = new char[17];
  int* new_state = reinterpret_cast<int*>(state);

  vector<int> symbols = {5, 10, 7};
  converter.convert(symbols, state);
  EXPECT_EQ(5, new_state[0]);
  EXPECT_EQ(10, new_state[1]);
  EXPECT_EQ(7, new_state[2]);
  EXPECT_EQ(3, state[16]);

  symbols = {5, 10, 7, 19};
  converter.convert(symbols, state);
  EXPECT_EQ(5, new_state[0]);
  EXPECT_EQ(10, new_state[1]);
  EXPECT_EQ(7, new_state[2]);
  EXPECT_EQ(19, new_state[3]);
  EXPECT_EQ(4, state[16]);


  delete state;
}

} // namespace oxlm
