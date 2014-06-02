#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/cdec_rule_converter.h"

namespace oxlm {

TEST(CdecRuleConverterTest, TestBasic) {
  Dict dict;
  for (int i = 0; i < 20; ++i) {
    dict.Convert(to_string(i));
  }
  boost::shared_ptr<CdecLBLMapper> mapper =
      boost::make_shared<CdecLBLMapper>(dict);
  boost::shared_ptr<CdecStateConverter> state_converter =
      boost::make_shared<CdecStateConverter>(16);
  CdecRuleConverter rule_converter(mapper, state_converter);

  char* state1 = new char[17];
  int* new_state1 = reinterpret_cast<int*>(state1);
  new_state1[0] = 5;
  new_state1[1] = 11;
  new_state1[2] = -1;
  new_state1[3] = 8;
  state1[16] = 4;

  char* state2 = new char[17];
  int* new_state2 = reinterpret_cast<int*>(state2);
  new_state2[0] = 1;
  new_state2[1] = -1;
  new_state2[2] = 3;
  state2[16] = 3;

  vector<int> target = {4, -1, 6, 5, 0};
  vector<const void*> states = {state1, state2};

  vector<int> expected_symbols = {3, 1, -1, 3, 5, 4, 5, 11, -1, 8};
  EXPECT_EQ(expected_symbols, rule_converter.convertTargetSide(target, states));

  delete state1;
  delete state2;
}

} // namespace oxlm
