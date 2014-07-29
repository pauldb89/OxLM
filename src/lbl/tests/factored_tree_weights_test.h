#pragma once

#include "lbl/factored_tree_weights.h"
#include "lbl/tests/weights_test.h"

namespace oxlm {

class FactoredTreeWeightsTest : public WeightsTest {
 protected:
  void SetUp() {
    WeightsTest::SetUp();
    config->vocab_size = 7;
    config->tree_file = "tree.txt";

    vector<int> data = {2, 3, 2, 4, 5, 6, 1};
    corpus = boost::make_shared<Corpus>(data);
    metadata = boost::make_shared<TreeMetadata>(config, vocab);
  }

  boost::shared_ptr<TreeMetadata> metadata;
};

} // namespace oxlm
