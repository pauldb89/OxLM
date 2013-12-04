#ifndef _PYPDICT_H_
#define _PYPDICT_H_

#include <string>
#include <iostream>
#include <cassert>
#include <fstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <functional>

#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/functional/hash.hpp>

namespace oxlm {

typedef std::string Word;
typedef int WordId;
typedef std::vector<WordId> Words;

class Dict {
// typedef std::unordered_map<std::string, WordId, std::hash<std::string> > Map;
// typedef std::map<std::string, WordId> Map;
 public:
  Dict() : b0_("<bad0>"), sos_("<s>"), eos_("</s>"), bad0_id_(-1) {
    words_.reserve(1000);
    Convert(sos_); Convert(eos_);
  }

  inline WordId min() const { return 0; }
  inline WordId max() const { return words_.size()-1; }
  inline size_t size() const { return words_.size(); }

  static bool is_ws(char x) {
    return (x == ' ' || x == '\t');
  }

  inline void ConvertWhitespaceDelimitedLine(const std::string& line, std::vector<WordId>* out) {
    size_t cur = 0;
    size_t last = 0;
    int state = 0;
    out->clear();
    while(cur < line.size()) {
      if (is_ws(line[cur++])) {
        if (state == 0) continue;
        out->push_back(Convert(line.substr(last, cur - last - 1)));
        state = 0;
      } else {
        if (state == 1) continue;
        last = cur - 1;
        state = 1;
      }
    }
    if (state == 1)
      out->push_back(Convert(line.substr(last, cur - last)));
  }

  inline WordId Lookup(const Word& word) const {
    auto i = d_.find(word);
    if (i == d_.end()) return bad0_id_;
    return i->second;
  }

  inline WordId Convert(const Word& word, bool frozen = false) {
    auto i = d_.find(word);
    if (i == d_.end()) {
      if (frozen)
        return bad0_id_;
      words_.push_back(word);
      d_[word] = words_.size()-1;
      return words_.size()-1;
    } else {
      return i->second;
    }
  }

  inline const std::vector<Word> getVocab() const {
    return words_;
  }
  
  inline bool valid(const WordId id) const { return id >= 0; }

  inline const Word& Convert(const WordId id) const {
    if (!valid(id)) return b0_;
    return words_[id];
  }

  friend class boost::serialization::access;
  template<class Archive> 
  void serialize(Archive & ar, const unsigned int version) {
    ar & b0_;
    ar & sos_;
    ar & eos_;
    ar & bad0_id_;
    ar & words_;
    ar & d_;
  }
 private:
  Word b0_, sos_, eos_;
  WordId bad0_id_;
  std::vector<Word> words_;
  std::map<std::string, WordId> d_;
};

inline void ReadFromFile(const std::string& filename,
                  Dict* d,
                  std::vector<std::vector<WordId> >* src,
                  std::set<WordId>* src_vocab) {
  src->clear();
  std::cerr << "Reading from " << filename << std::endl;
  std::ifstream in(filename);
  assert(in);
  std::string line;
  int lc = 0;
  while(getline(in, line)) {
    ++lc;
    src->push_back(std::vector<WordId>());
    d->ConvertWhitespaceDelimitedLine(line, &src->back());
    for (WordId i = 0; i < static_cast<WordId>(src->back().size()); ++i) src_vocab->insert(src->back()[i]);
  }
}

template <typename Container> 
struct container_hash {
  std::size_t operator()(Container const& c) const {
    return boost::hash_range(c.begin(), c.end());
  }
};

}


namespace std {
  template<typename S, typename T> struct hash<pair<S, T>> {
    inline size_t operator()(const pair<S, T> & v) const {
      size_t seed = 0;
      boost::hash_combine(seed, v.first);
      boost::hash_combine(seed, v.second);
      return seed;
    }
  };
}


#endif // PYPDICT_H_
