#ifndef _CPYP_SPARSE_VECTOR_H_
#define _CPYP_SPARSE_VECTOR_H_

// SparseVector<T> is a integer indexed unordered map that supports very fast
// (mathematical) vector operations when the sizes are very small, and reasonably
// fast operations when the sizes are large.
// important: indexes are integers
// important: iterators may return elements in any order

#include <cmath>
#include <cstring>
#include <climits>
#include <unordered_map>
#include <cassert>
#include <vector>
#include <limits>

namespace cpyp {

// this should just be a typedef to pair<unsigned,T> on the new c++
// I have to avoid this since I want to use unions and c++-98
// does not let unions have types with constructors in them
// this type bypasses default constructors. use with caution!
// this should work as long as T does have a destructor that
// does anything
template <typename T>
struct PairIntT {
  const PairIntT& operator=(const std::pair<const unsigned, T>& v) {
    std::memcpy(this, &v, sizeof(PairIntT));
    return *this;
  }
  operator const std::pair<const unsigned, T>&() const {
    return *reinterpret_cast<const std::pair<const unsigned, T>*>(this);
  }
  unsigned& first() {
    return reinterpret_cast<std::pair<unsigned, T>*>(this)->first;
  }
  T& second() {
    return reinterpret_cast<std::pair<unsigned, T>*>(this)->second;
  }
  const unsigned& first() const {
    return reinterpret_cast<const std::pair<unsigned, T>*>(this)->first;
  }
  const T& second() const {
    return reinterpret_cast<const std::pair<unsigned, T>*>(this)->second;
  }
 private:
  // very bad way of bypassing the default constructor on T
  char data_[sizeof(std::pair<unsigned, T>)];
};

template <typename T, unsigned LOCAL_MAX = (sizeof(T) == sizeof(float) ? 15u : 7u)>
class SparseVector {
 public:
  struct iterator {
    iterator(SparseVector<T>& v, const bool is_end) : local_(!v.is_remote_) {
      if (local_) {
        local_it_ = &v.data_.local[is_end ? v.local_size_ : 0];
      } else {
        if (is_end)
          remote_it_ = v.data_.rbmap->end();
        else
          remote_it_ = v.data_.rbmap->begin();
      }
    }
    iterator(SparseVector<T>& v, const bool, const unsigned k) : local_(!v.is_remote_) {
      if (local_) {
        unsigned i = 0;
        while(i < v.local_size_ && v.data_.local[i].first() != k) { ++i; }
        local_it_ = &v.data_.local[i];
      } else {
        remote_it_ = v.data_.rbmap->find(k);
      }
    }
    const bool local_;
    PairIntT<T>* local_it_;
    typename std::unordered_map<unsigned, T>::iterator remote_it_;
    std::pair<const unsigned, T>& operator*() const {
      if (local_)
        return *reinterpret_cast<std::pair<const unsigned, T>*>(local_it_);
      else
        return *remote_it_;
    }

    std::pair<const unsigned, T>* operator->() const {
      if (local_)
        return reinterpret_cast<std::pair<const unsigned, T>*>(local_it_);
      else
        return &*remote_it_;
    }

    iterator& operator++() {
      if (local_) ++local_it_; else ++remote_it_;
      return *this;
    }

    inline bool operator==(const iterator& o) const {
      if (o.local_ != local_) return false;
      if (local_) {
        return local_it_ == o.local_it_;
      } else {
        return remote_it_ == o.remote_it_;
      }
    }
    inline bool operator!=(const iterator& o) const {
      return !(o == *this);
    }
  };
  struct const_iterator {
    const_iterator(const SparseVector<T>& v, const bool is_end) : local_(!v.is_remote_) {
      if (local_) {
        local_it_ = &v.data_.local[is_end ? v.local_size_ : 0];
      } else {
        if (is_end)
          remote_it_ = v.data_.rbmap->end();
        else
          remote_it_ = v.data_.rbmap->begin();
      }
    }
    const_iterator(const SparseVector<T>& v, const bool, const unsigned k) : local_(!v.is_remote_) {
      if (local_) {
        unsigned i = 0;
        while(i < v.local_size_ && v.data_.local[i].first() != k) { ++i; }
        local_it_ = &v.data_.local[i];
      } else {
        remote_it_ = v.data_.rbmap->find(k);
      }
    }
    const bool local_;
    const PairIntT<T>* local_it_;
    typename std::unordered_map<unsigned, T>::const_iterator remote_it_;
    const std::pair<const unsigned, T>& operator*() const {
      if (local_)
        return *reinterpret_cast<const std::pair<const unsigned, T>*>(local_it_);
      else
        return *remote_it_;
    }

    const std::pair<const unsigned, T>* operator->() const {
      if (local_)
        return reinterpret_cast<const std::pair<const unsigned, T>*>(local_it_);
      else
        return &*remote_it_;
    }

    const_iterator& operator++() {
      if (local_) ++local_it_; else ++remote_it_;
      return *this;
    }

    inline bool operator==(const const_iterator& o) const {
      if (o.local_ != local_) return false;
      if (local_) {
        return local_it_ == o.local_it_;
      } else {
        return remote_it_ == o.remote_it_;
      }
    }
    inline bool operator!=(const const_iterator& o) const {
      return !(o == *this);
    }
  };
 public:
  SparseVector() : local_size_(0), is_remote_(false) { std::memset(&data_, 0, sizeof(data_)); }
  ~SparseVector() {
    clear();
  }
  SparseVector(const SparseVector& other) {
    std::memcpy(this, &other, sizeof(SparseVector));
    if (is_remote_) data_.rbmap = new std::unordered_map<unsigned, T>(*data_.rbmap);
  }
  SparseVector(std::pair<unsigned, T>* first, std::pair<unsigned, T>* last) {
    const unsigned n = last - first;
    if (n <= LOCAL_MAX) {
      is_remote_ = false;
      local_size_ = n;
      std::memcpy(data_.local, first, sizeof(std::pair<unsigned, T>) * n);
    } else {
      is_remote_ = true;
      data_.rbmap = new std::unordered_map<unsigned, T>(first, last);
    }
  }
  void erase(unsigned k) {
    if (is_remote_) {
      data_.rbmap->erase(k);
    } else {
      for (unsigned i = 0; i < local_size_; ++i) {
        if (data_.local[i].first() == k) {
          for (unsigned j = i+1; j < local_size_; ++j) {
            data_.local[j-1].first() = data_.local[j].first();
            data_.local[j-1].second() = data_.local[j].second();
          }
        }
      }
      --local_size_;
    }
  }
  const SparseVector<T>& operator=(const SparseVector<T>& other) {
    if (&other == this) return *this;
    clear();
    std::memcpy(this, &other, sizeof(SparseVector));
    if (is_remote_) {
      data_.rbmap = new std::unordered_map<unsigned, T>(*data_.rbmap);
    }
    return *this;
  }
  T const& get_singleton() const {
    assert(size()==1);
    return begin()->second;
  }
  bool nonzero(unsigned k) const {
    return static_cast<bool>(value(k));
  }
  inline T& operator[](unsigned k) {
    return get_or_create_bin(k);
  }
  inline void set_value(unsigned k, const T& v) {
    get_or_create_bin(k) = v;
  }
  inline T& add_value(unsigned k, const T& v) {
    return get_or_create_bin(k) += v;
  }
  inline T get(unsigned k) const {
    return value(k);
  }
  inline T value(unsigned k) const {
    if (is_remote_) {
      typename std::unordered_map<unsigned, T>::const_iterator it = data_.rbmap->find(k);
      if (it != data_.rbmap->end()) return it->second;
    } else {
      for (unsigned i = 0; i < local_size_; ++i) {
        const PairIntT<T>& p = data_.local[i];
        if (p.first() == k) return p.second();
      }
    }
    return T();
  }
  T l2norm_sq() const {
    T sum = T();
    for (const_iterator it = begin(), e = end(); it != e; ++it)
      sum += it->second * it->second;
    return sum;
  }
  T l2norm() const {
    return sqrt(l2norm_sq());
  }
  T pnorm(const double p) const {
    T sum = T();
    for (const_iterator it = begin(), e = end(); it != e; ++it)
      sum += pow(fabs(it->second), p);
    return pow(sum, 1.0 / p);
  }
  // if values are binary, gives |A intersect B|/|A union B|
  template<typename S>
  S tanimoto_coef(const SparseVector<S> &vec) const {
    const S dp=dot(vec);
    return dp/(l2norm_sq()+vec.l2norm_sq()-dp);
  }
  inline size_t size() const {
    if (is_remote_)
      return data_.rbmap->size();
    else
      return local_size_;
  }
  size_t num_nonzero() const {
    size_t sz = 0;
    const_iterator it = this->begin();
    for (; it != this->end(); ++it) {
      if (nonzero(it->first)) sz++; 
    }
    return sz;
  }
  inline void clear() {
    if (is_remote_) delete data_.rbmap;
    is_remote_ = false;
    local_size_ = 0;
  }
  inline bool empty() const {
    return size() == 0;
  }
  inline SparseVector& operator+=(const SparseVector& other) {
    if (empty()) { *this = other; return *this; }
    const typename SparseVector::const_iterator end = other.end();
    for (typename SparseVector::const_iterator it = other.begin(); it != end; ++it) {
      get_or_create_bin(it->first) += it->second;
    }
    return *this;
  }
  template <typename O>
  inline SparseVector& operator+=(const SparseVector<O>& other) {
    const typename SparseVector<O>::const_iterator end = other.end();
    for (typename SparseVector<O>::const_iterator it = other.begin(); it != end; ++it) {
      get_or_create_bin(it->first) += it->second;
    }
    return *this;
  }
  template <typename O>
  inline void plus_eq_v_times_s(const SparseVector<O>& other, const O scalar) {
    const typename SparseVector<O>::const_iterator end = other.end();
    for (typename SparseVector<O>::const_iterator it = other.begin(); it != end; ++it) {
      get_or_create_bin(it->first) += it->second * scalar;
    }
  }
  inline SparseVector& operator-=(const SparseVector& other) {
    const typename SparseVector::const_iterator end = other.end();
    for (typename SparseVector::const_iterator it = other.begin(); it != end; ++it) {
      get_or_create_bin(it->first) -= it->second;
    }
    return *this;
  }
  inline SparseVector& operator*=(const T& scalar) {
    if (is_remote_) {
      const typename std::unordered_map<unsigned, T>::iterator end = data_.rbmap->end();
      for (typename std::unordered_map<unsigned, T>::iterator it = data_.rbmap->begin(); it != end; ++it)
        it->second *= scalar;
    } else {
      for (int i = 0; i < local_size_; ++i)
        data_.local[i].second() *= scalar;
    }
    return *this;
  }
  inline SparseVector& operator/=(const T& scalar) {
    if (is_remote_) {
      const typename std::unordered_map<unsigned, T>::iterator end = data_.rbmap->end();
      for (typename std::unordered_map<unsigned, T>::iterator it = data_.rbmap->begin(); it != end; ++it)
        it->second /= scalar;
    } else {
      for (int i = 0; i < local_size_; ++i)
        data_.local[i].second() /= scalar;
    }
    return *this;
  }
  SparseVector<T> erase_zeros(const T& EPSILON = 1e-4) const {
    SparseVector<T> o;
    for (const_iterator it = begin(); it != end(); ++it) {
      if (fabs(it->second) > EPSILON) o.set_value(it->first, it->second);
    }
    return o;
  }
  iterator find(unsigned k) {
    return iterator(*this, false, k);
  }
  iterator begin() {
    return iterator(*this, false);
  }
  iterator end() {
    return iterator(*this, true);
  }
  const_iterator find(unsigned k) const {
    return const_iterator(*this, false, k);
  }
  const_iterator begin() const {
    return const_iterator(*this, false);
  }
  const_iterator end() const {
    return const_iterator(*this, true);
  }
  void init_vector(std::vector<T> *vp) const {
    init_vector(*vp);
  }
  void init_vector(std::vector<T> &v) const {
    v.clear();
    for (const_iterator i=begin(),e=end();i!=e;++i)
      extend_vector(v,i->first)=i->second;
  }
  T dot(const std::vector<T>& v) const {
    T res = T();
    for (const_iterator it = begin(), e = end(); it != e; ++it)
      if (static_cast<unsigned>(it->first) < v.size()) res += it->second * v[it->first];
    return res;
  }
  T dot(const SparseVector<T>& other) const {
    T res = T();
    for (const_iterator it = begin(), e = end(); it != e; ++it)
      res += other.value(it->first) * it->second;
    return res;
  }
  bool operator==(const SparseVector<T>& other) const {
    if (other.size() != size()) return false;
    for (const_iterator it = begin(), e = end(); it != e; ++it) {
      if (other.value(it->first) != it->second) return false;
    }
    return true;
  }
  void swap(SparseVector<T>& other) {
    char t[sizeof(data_)];
    std::swap(other.is_remote_, is_remote_);
    std::swap(other.local_size_, local_size_);
    std::memcpy(t, &other.data_, sizeof(data_));
    std::memcpy(&other.data_, &data_, sizeof(data_));
    std::memcpy(&data_, t, sizeof(data_));
  }
  template<class Archive>
  void save(Archive & ar, const unsigned int) const {
    int eff_size = size();
    const_iterator it = this->begin();
    ar & eff_size;
    while (it != this->end()) {
      ar & *it;
      ++it;
    }
  }
  template<class Archive>
  void load(Archive & ar, const unsigned int) {
    this->clear();
    unsigned sz; ar & sz;
    for (unsigned i = 0; i < sz; ++i) {
      std::pair<unsigned, T> wire_pair;
      ar & wire_pair;
      this->set_value(wire_pair.first, wire_pair.second);
    }
  }
 private:
  static inline T& extend_vector(std::vector<T> &v,size_t i) {
    if (i>=v.size())
      v.resize(i+1);
    return v[i];
  }
  inline T& get_or_create_bin(unsigned k) {
    if (is_remote_) {
      return (*data_.rbmap)[k];
    } else {
      for (unsigned i = 0; i < local_size_; ++i)
        if (data_.local[i].first() == k) return data_.local[i].second();
    }
    assert(!is_remote_);
    // currently local!
    if (local_size_ < LOCAL_MAX) {
      PairIntT<T>& p = data_.local[local_size_];
      ++local_size_;
      p.first() = k;
      p.second() = T();
      return p.second();
    } else {
      swap_local_rbmap();
      return (*data_.rbmap)[k];  
    }
  }
  void swap_local_rbmap() {
    if (is_remote_) { // data is in rbmap, move to local
      assert(data_.rbmap->size() < LOCAL_MAX);
      const std::unordered_map<unsigned, T>* m = data_.rbmap;
      local_size_ = m->size();
      int i = 0;
      for (typename std::unordered_map<unsigned, T>::const_iterator it = m->begin();
           it != m->end(); ++it) {
        data_.local[i] = *it;
        ++i;
      }
      is_remote_ = false;
    } else { // data is local, move to rbmap
      std::unordered_map<unsigned, T>* m = new std::unordered_map<unsigned, T>(
         reinterpret_cast<std::pair<unsigned, T>*>(&data_.local[0]),
         reinterpret_cast<std::pair<unsigned, T>*>(&data_.local[local_size_]), local_size_ * 1.5 + 1);
      data_.rbmap = m;
      is_remote_ = true;
    }
  }

  union {
    PairIntT<T> local[LOCAL_MAX];
    std::unordered_map<unsigned, T>* rbmap;
  } data_;
  unsigned char local_size_;
  bool is_remote_;
};

template <typename T>
void swap(SparseVector<T>& a, SparseVector<T>& b) {
  a.swap(b);
}

template <typename T>
const SparseVector<T> operator+(const SparseVector<T>& x, const SparseVector<T>& y) {
  if (x.size() > y.size()) {
    SparseVector<T> res(x);
    res += y;
    return res;
  } else {
    SparseVector<T> res(y);
    res += x;
    return res;
  }
}

template <typename T>
const SparseVector<T> operator-(const SparseVector<T>& x, const SparseVector<T>& y) {
  SparseVector<T> res(x);
  res -= y;
  return res;
}

template <class T>
std::size_t hash_value(SparseVector<T> const& x) {
  assert(!"not implemented");
  return 0;
}

template <class T, typename S>
SparseVector<T> operator*(const SparseVector<T>& a, const S& b) {
  SparseVector<T> result = a;
  return result *= b;
}

template <class T>
SparseVector<T> operator*(const SparseVector<T>& a, const double& b) {
  SparseVector<T> result = a;
  return result *= b;
}

template <class T, typename S>
SparseVector<T> operator/(const SparseVector<T>& a, const S& b) {
  SparseVector<T> result = a;
  return result /= b;
}

template <class T>
SparseVector<T> operator/(const SparseVector<T>& a, const double& b) {
  SparseVector<T> result = a;
  return result /= b;
}

}

#endif
