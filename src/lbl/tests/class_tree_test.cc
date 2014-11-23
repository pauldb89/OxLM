#include "gtest/gtest.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/make_shared.hpp>

#include "lbl/class_tree.h"

namespace ar = boost::archive;

namespace oxlm {

TEST(ClassTreeTest, TestBasic) {
  boost::shared_ptr<Vocabulary> vocab = boost::make_shared<Vocabulary>();
  ClassTree tree("tree.txt", vocab);

  EXPECT_EQ(7, vocab->size());

  vector<string> words = {"<s>", "</s>", "a", "b", "c", "d", "e"};
  for (size_t i = 0; i < words.size(); ++i) {
    EXPECT_EQ(i, vocab->convert(words[i]));
    EXPECT_EQ(words[i], vocab->convert(i));
  }

  EXPECT_EQ(12, tree.size());

  vector<int> expected_parent = {-1, 0, 0, 2, 2, 3, 3, 4, 4, 4, 5, 5};
  for (size_t i = 0; i < tree.size(); ++i) {
    EXPECT_EQ(expected_parent[i], tree.getParent(i));
  }

  vector<int> expected_children = {1, 2};
  EXPECT_EQ(expected_children, tree.getChildren(0));
  expected_children = {};
  EXPECT_EQ(expected_children, tree.getChildren(1));
  expected_children = {3, 4};
  EXPECT_EQ(expected_children, tree.getChildren(2));
  expected_children = {5, 6};
  EXPECT_EQ(expected_children, tree.getChildren(3));
  expected_children = {7, 8, 9};
  EXPECT_EQ(expected_children, tree.getChildren(4));
  expected_children = {10, 11};
  EXPECT_EQ(expected_children, tree.getChildren(5));
  expected_children = {};
  EXPECT_EQ(expected_children, tree.getChildren(6));
  expected_children = {};
  EXPECT_EQ(expected_children, tree.getChildren(7));
  expected_children = {};
  EXPECT_EQ(expected_children, tree.getChildren(8));
  expected_children = {};
  EXPECT_EQ(expected_children, tree.getChildren(9));
  expected_children = {};
  EXPECT_EQ(expected_children, tree.getChildren(10));
  expected_children = {};
  EXPECT_EQ(expected_children, tree.getChildren(11));

  EXPECT_EQ(1, tree.getNode(0));
  EXPECT_EQ(10, tree.getNode(1));
  EXPECT_EQ(11, tree.getNode(2));
  EXPECT_EQ(6, tree.getNode(3));
  EXPECT_EQ(7, tree.getNode(4));
  EXPECT_EQ(8, tree.getNode(5));
  EXPECT_EQ(9, tree.getNode(6));

  EXPECT_EQ(0, tree.childIndex(1));
  EXPECT_EQ(1, tree.childIndex(2));
  EXPECT_EQ(0, tree.childIndex(3));
  EXPECT_EQ(1, tree.childIndex(4));
  EXPECT_EQ(0, tree.childIndex(5));
  EXPECT_EQ(1, tree.childIndex(6));
  EXPECT_EQ(0, tree.childIndex(7));
  EXPECT_EQ(1, tree.childIndex(8));
  EXPECT_EQ(2, tree.childIndex(9));
  EXPECT_EQ(0, tree.childIndex(10));
  EXPECT_EQ(1, tree.childIndex(11));
}

TEST(ClassTreeTest, TestSerialization) {
  boost::shared_ptr<Vocabulary> vocab = boost::make_shared<Vocabulary>();
  ClassTree tree("tree.txt", vocab), tree_copy;

  stringstream stream(ios_base::in | ios_base::out | ios_base::binary);
  ar::binary_oarchive oar(stream, ar::no_header);
  oar << tree;

  ar::binary_iarchive iar(stream, ar::no_header);
  iar >> tree_copy;

  EXPECT_EQ(tree, tree_copy);
}

} // namespace oxlm
