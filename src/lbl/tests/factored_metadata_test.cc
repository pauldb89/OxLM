#include "gtest/gtest.h"

#include "lbl/factored_metadata.h"
#include "utils/constants.h"

namespace oxlm {

TEST(FactoredMetadataTest, TestBasic) {
  ModelData config;
  config.training_file = "training.txt";
  config.class_file = "classes.txt";
  Dict dict;
  FactoredMetadata metadata(config, dict);

  EXPECT_NEAR(1, metadata.getClassBias().array().exp().sum(), EPS);
  EXPECT_EQ(36, metadata.getIndex()->getNumClasses());
}

} // namespace oxlm
