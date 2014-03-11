#include "lbl/utils.h"

namespace oxlm {

Time GetTime() {
  return Clock::now();
}

double GetDuration(const Time& start_time, const Time& stop_time) {
  return duration_cast<milliseconds>(stop_time - start_time).count() / 1000.0;
}

} // namespace oxlm
