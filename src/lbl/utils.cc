#include "lbl/utils.h"

#include <unordered_set>

#include <boost/algorithm/string.hpp>

using namespace boost::algorithm;

namespace oxlm {

Time GetTime() {
  return Clock::now();
}

double GetDuration(const Time& start_time, const Time& stop_time) {
  return duration_cast<milliseconds>(stop_time - start_time).count() / 1000.0;
}

void printMemoryUsage() {
  unordered_set<string> headers = {"VmPeak:", "VmRSS:"};
  ifstream metadata("/proc/self/status", ios::in);
  string header, value;
  while ((metadata >> header) && getline(metadata, value)) {
    if (headers.count(header)) {
      trim(value);
      cout << header << " " << value << endl;
    }
  }
}

} // namespace oxlm
