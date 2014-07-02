#include "gtest/gtest.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/make_shared.hpp>

#include "lbl/factored_metadata.h"
#include "utils/constants.h"

namespace ar = boost::archive;

namespace oxlm {

TEST(FactoredMetadataTest, TestBasic) {
  boost::shared_ptr<ModelData> config = boost::make_shared<ModelData>();
  config->training_file = "training.txt";
  config->class_file = "classes.txt";
  Dict dict;
  FactoredMetadata metadata(config, dict);

  EXPECT_NEAR(1, metadata.getClassBias().array().exp().sum(), EPS);
  EXPECT_EQ(36, metadata.getIndex()->getNumClasses());
}

TEST(FactoredMetadataTest, TestSerialization) {
  boost::shared_ptr<ModelData> config = boost::make_shared<ModelData>();
  config->training_file = "training.txt";
  config->class_file = "classes.txt";
  Dict dict;
  FactoredMetadata metadata(config, dict), metadata_copy;

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << metadata;

  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> metadata_copy;

  EXPECT_EQ(metadata, metadata_copy);
}

} // namespace oxlm
