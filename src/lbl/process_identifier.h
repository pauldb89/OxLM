#pragma once

#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
// boost::unordered_set is compatible with boost::interprocess, unlike
// std::unordered_set.
#include <boost/unordered_set.hpp>

using namespace std;
namespace ip = boost::interprocess;

namespace oxlm {

typedef ip::managed_shared_memory SharedMemory;
typedef ip::allocator<int, SharedMemory::segment_manager> ShMemAllocator;
typedef boost::unordered_set<int, boost::hash<int>, equal_to<int>, ShMemAllocator> ProcessIdSet;
typedef ip::interprocess_mutex SharedMutex;


class ProcessIdentifier {
 public:
  ProcessIdentifier(const char* segment_name);

  int getId();

  void freeId(int process_id);

 private:
  static const int MAX_PROCESS_ID;

  SharedMemory segment;
  ProcessIdSet* processIds;
  SharedMutex* mutex;
};

} // namespace oxlm
