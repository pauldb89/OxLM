#include "gtest/gtest.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "lbl/tree_metadata.h"
#include "utils/constants.h"
#include "utils/testing.h"

namespace ar = boost::archive;

namespace oxlm {

TEST(TreeMetadataTest, TestBasic) {
  boost::shared_ptr<ModelData> config = boost::make_shared<ModelData>();
  config->tree_file = "tree.txt";
  boost::shared_ptr<Vocabulary> vocab;
  TreeMetadata metadata(config, vocab);

  vector<int> data = {2, 3, 2, 4, 1};
  boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(data);
  metadata.initialize(corpus);
  VectorReal expected_unigram = VectorReal::Zero(7);
  expected_unigram << 0, 0.2, 0.4, 0.2, 0.2, 0, 0;
  EXPECT_MATRIX_NEAR(expected_unigram, metadata.getUnigram(), EPS);

  boost::shared_ptr<ClassTree> tree = metadata.getTree();
  EXPECT_EQ(12, tree->size());

  vector<int> expected_nodes = {1, 10, 11, 6, 7, 8, 9};
  for (int i = 0; i < config->vocab_size; ++i) {
    EXPECT_EQ(expected_nodes[i], tree->getNode(i));
  }

  vector<int> expected_parent = {-1, 0, 0, 2, 2, 3, 3, 4, 4, 4, 5, 5};
  for (size_t i = 0; i < tree->size(); ++i) {
    EXPECT_EQ(expected_parent[i], tree->getParent(i));
  }

  vector<int> expected_child_index = {0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 1};
  for (size_t i = 1; i < tree->size(); ++i) {
    EXPECT_EQ(expected_child_index[i - 1], tree->childIndex(i));
  }
}

TEST(TreeMetadataTest, TestSerialization) {
  boost::shared_ptr<ModelData> config = boost::make_shared<ModelData>();
  config->tree_file = "tree.txt";
  boost::shared_ptr<Vocabulary> vocab;
  TreeMetadata metadata(config, vocab), metadata_copy;

  stringstream stream(ios_base::binary | ios_base::in | ios_base::out);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << metadata;

  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> metadata_copy;

  EXPECT_EQ(metadata, metadata_copy);
}

} // namespace oxlm
