#ifndef UVECTOR_H_
#define UVECTOR_H_

#include <vector>

struct uvector_hash {
  size_t operator()(const std::vector<unsigned>& v) const {
    size_t h = v.size();
    for (auto e : v)
      h ^= e + 0x9e3779b9 + (h<<6) + (h>>2);
    return h;
  }
};

struct vector_hash {
  size_t operator()(const std::vector<int>& v) const {
    size_t h = v.size();
    for (auto e : v)
      h ^= (size_t)e + 0x9e3779b9 + (h<<6) + (h>>2);
    return h;
  }
};

#endif
