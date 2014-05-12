#include "lbl/process_identifier.h"

#include <boost/interprocess/sync/scoped_lock.hpp>

namespace oxlm {

const int ProcessIdentifier::MAX_PROCESS_ID = 1000;

ProcessIdentifier::ProcessIdentifier(const char* segment_name) {
  segment = SharedMemory(ip::open_or_create, segment_name, 1 << 16);
  processIds = segment.find_or_construct<ProcessIdSet>("ProcessIds")
      (0, boost::hash<int>(), equal_to<int>(), segment.get_allocator<int>());
  mutex = segment.find_or_construct<SharedMutex>("Mutex")();
}

int ProcessIdentifier::reserveId() {
  ip::scoped_lock<SharedMutex> lock(*mutex);

  int process_id;
  for (process_id = 0; process_id < MAX_PROCESS_ID; ++process_id) {
    if (!processIds->count(process_id)) {
      break;
    }
  }

  assert(process_id < MAX_PROCESS_ID);
  processIds->insert(process_id);

  return process_id;
}

void ProcessIdentifier::freeId(int process_id) {
  ip::scoped_lock<SharedMutex> lock(*mutex);
  processIds->erase(process_id);
}

} // namespace oxlm
