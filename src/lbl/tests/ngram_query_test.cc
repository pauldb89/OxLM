#include "gtest/gtest.h"

#include "lbl/ngram_query.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace ar = boost::archive;

namespace oxlm {

TEST(NGramQueryTest, TestSerialization) {
  vector<int> context = {2, 3, 4, 5};
  NGramQuery query(1, context);

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << query;

  NGramQuery query_copy;
  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> query_copy;

  EXPECT_EQ(query, query_copy);
}

} // namespace oxlm
