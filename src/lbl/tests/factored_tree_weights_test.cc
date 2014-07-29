#include "lbl/tests/factored_tree_weights_test.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "utils/constants.h"

namespace ar = boost::archive;

namespace oxlm {

TEST_F(FactoredTreeWeightsTest, TestCheckGradient) {
  FactoredTreeWeights weights(config, metadata, corpus);
  vector<int> indices = {0, 1, 2, 3, 4, 5, 6};
  Real log_likelihood;
  MinibatchWords words;
  boost::shared_ptr<FactoredTreeWeights> gradient =
      boost::make_shared<FactoredTreeWeights>(config, metadata);
  weights.getGradient(corpus, indices, gradient, log_likelihood, words);

  // See comment in weights_test.
  EXPECT_TRUE(weights.checkGradient(corpus, indices, gradient, 1e-3));
}

TEST_F(FactoredTreeWeightsTest, TestSerialization) {
  FactoredTreeWeights weights(config, metadata, corpus), weights_copy;

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << weights;

  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> weights_copy;

  EXPECT_EQ(weights, weights_copy);
}

} // namespace oxlm
