#include "gtest/gtest.h"

#include "lbl/query_cache.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace ar = boost::archive;

namespace oxlm {

TEST(QueryCacheTest, TestBasic) {
  QueryCache cache;
  EXPECT_EQ(0, cache.size());

  vector<int> context = {2, 3, 4, 5};
  NGram query(1, context);
  EXPECT_EQ(make_pair(Real(0.0), false), cache.get(query));

  cache.put(query, 0.5);
  EXPECT_EQ(1, cache.size());
  EXPECT_EQ(make_pair(Real(0.5), true), cache.get(query));

  cache.put(query, 0.7);
  EXPECT_EQ(1, cache.size());
  EXPECT_EQ(make_pair(Real(0.7), true), cache.get(query));
}

TEST(QueryCacheTest, TestSerialization) {
  QueryCache cache;
  vector<int> context = {2, 3, 4, 5};
  NGram query(1, context);
  cache.put(query, 0.5);

  stringstream stream(ios_base::binary | ios_base::out | ios_base::in);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << cache;

  QueryCache cache_copy;
  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> cache_copy;

  EXPECT_EQ(cache, cache_copy);
  EXPECT_EQ(1, cache_copy.size());
  EXPECT_EQ(make_pair(Real(0.5), true), cache_copy.get(query));
}

} // namespace olxm
