#include "gtest/gtest.h"

#include "lbl/process_identifier.h"

namespace oxlm {

TEST(ProcessIdentifierTest, TestBasic) {
  ProcessIdentifier process_identifier("ProcessIdentifierTest");
  int ret1 = process_identifier.getId();
  int ret2 = process_identifier.getId();
  int ret3 = process_identifier.getId();
  process_identifier.freeId(1);
  int ret4 = process_identifier.getId();

  // Clean up before checks.
  process_identifier.freeId(0);
  process_identifier.freeId(1);
  process_identifier.freeId(2);

  EXPECT_EQ(0, ret1);
  EXPECT_EQ(1, ret2);
  EXPECT_EQ(2, ret3);
  EXPECT_EQ(1, ret4);
}

TEST(ProcessIdentifierTest, TestMultithreaded) {
  vector<int> process_ids;
  #pragma omp parallel num_threads(2)
  {
    ProcessIdentifier process_identifier("ProcessIdentifierTest");
    for (int i = 0; i < 100; ++i) {
      #pragma omp critical
      process_ids.push_back(process_identifier.getId());
    }

    // Clean up before checks. Wait for the first part to finish.
    #pragma omp barrier
    for (int i = 0; i < 200; ++i) {
      process_identifier.freeId(i);
    }
  }

  sort(process_ids.begin(), process_ids.end());
  for (int i = 0; i < 200; ++i) {
    EXPECT_EQ(i, process_ids[i]);
  }
}

} // namespace oxlm
