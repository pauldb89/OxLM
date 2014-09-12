#include "gtest/gtest.h"

#include <boost/make_shared.hpp>

#include "lbl/cdec_lbl_mapper.h"

#include "hg.h"

namespace oxlm {

TEST(CdecLBLMapperTest, TestBasic) {
  boost::shared_ptr<Vocabulary> vocab = boost::make_shared<Vocabulary>();
  vocab->convert("<s>");
  vocab->convert("</s>");
  vocab->convert("foo");
  vocab->convert("bar");
  CdecLBLMapper mapper(vocab);

  EXPECT_EQ(0, mapper.convert(1));
  EXPECT_EQ(1, mapper.convert(2));
  EXPECT_EQ(2, mapper.convert(3));
  EXPECT_EQ(3, mapper.convert(4));
}

} // namespace oxlm
