#ifndef _CPYP_CRP_TABLE_MANAGER_H_
#define _CPYP_CRP_TABLE_MANAGER_H_

#include <iostream>
#include <utility>
#include "msparse_vector.h"
#include "random.h"

namespace pyp {

// these are helper classes for implementing token-based CRP samplers
// basically the data structures recommended by Blunsom et al. in the Note.
// they are extended to deal with multifloor CRPs (see Wood & Teh, 2009)
// but if you don't care about this, just set the number of floors to 1
struct crp_histogram {
  //typedef std::map<unsigned, unsigned> MAPTYPE;
  typedef SparseVector<unsigned> MAPTYPE;
  typedef MAPTYPE::const_iterator const_iterator;

  inline void increment(unsigned bin, unsigned delta = 1u) {
    data[bin] += delta;
  }
  inline void decrement(unsigned bin, unsigned delta = 1u) {
    unsigned r = data[bin] -= delta;
    if (!r) data.erase(bin);
  }
  inline void move(unsigned from_bin, unsigned to_bin, unsigned delta = 1u) {
    decrement(from_bin, delta);
    increment(to_bin, delta);
  }
  bool empty() const { return data.empty(); }
  inline const_iterator begin() const { return data.begin(); }
  inline const_iterator end() const { return data.end(); }
  void swap(crp_histogram& other) {
    std::swap(data, other.data);
  }

  template<class Archive> void serialize(Archive& ar, const unsigned int version) {
    ar & data;
  }
 private:
  MAPTYPE data;
};

void swap(crp_histogram& a, crp_histogram& b) {
  a.swap(b);
}

// A crp_table_manager tracks statistics about all customers
// and tables serving some dish in a CRP, as well as what
// floor of the restaurant they are on (for multifloor variants of
// the CRP). it knows how to correctly sample what table to remove
// a customer from and what table to join
template <unsigned NumFloors>
struct crp_table_manager {
  crp_table_manager() : customers(), tables() {}

  inline unsigned num_tables() const {
    return tables;
  }

  inline unsigned num_customers() const {
    return customers;
  }

  inline void create_table(const unsigned floor = 0) {
    assert(floor < NumFloors);
    h[floor].increment(1);
    ++tables;
    ++customers;
  }

  // seat a customer at a table proportional to the number of customers seated at a table, less the discount
  // *new tables are never created by this function!
  // returns the number of customers already seated at the table (always > 0)
  template<typename Engine>
  unsigned share_table(const double discount, Engine& eng) {
    const double z = customers - discount * num_tables();
    double r = z * sample_uniform01<double>(eng);
    const auto floor_count = [&]()->std::pair<unsigned,int> {
      for (unsigned floor = 0; floor < NumFloors; ++floor) {
        const auto end = h[floor].end();
        auto i = h[floor].begin();
        for (; i != end; ++i) {
          const double thresh = (i->first - discount) * i->second;
          if (thresh > r) return std::make_pair(floor, i->first);
          r -= thresh;
        }
      }
      std::cerr << "Serious error while incrementing: Floors=" << NumFloors
                << " r=" << r << std::endl;
      std::abort();
    }();
    const unsigned cc = floor_count.second;
    h[floor_count.first].move(cc, cc + 1);
    ++customers;
    return cc;
  }

  // randomly sample a customer
  // *tables may be removed
  // returns (floor,table delta). Will be (0,0) unless a table is removed
  template<typename Engine>
  inline std::pair<unsigned,int> remove_customer(Engine& eng, unsigned* selected_table_postcount) {
    int r = sample_uniform01<double>(eng) * num_customers();
    const auto floor_count = [&]()->std::pair<unsigned,int> {
      for (unsigned floor = 0; floor < NumFloors; ++floor) {
        const auto end = h[floor].end();
        auto i = h[floor].begin();
        for (; i != end; ++i) {
          // sample randomly, i.e. *don't* discount
          const int thresh = i->first * i->second;
          if (thresh > r) return std::make_pair(floor, i->first);
          r -= thresh;
        }
      }
      std::cerr << "Serious error while decrementing: Floors=" << NumFloors
                << " r=" << r << std::endl;
      std::abort();
    }();
    --customers;
    const unsigned tc = floor_count.second;
    if (selected_table_postcount) *selected_table_postcount = tc - 1;
    if (tc == 1) { // remove customer from table containing a single customer
      h[floor_count.first].decrement(1);
      --tables;
      return std::make_pair(floor_count.first, -1);
    } else {
      h[floor_count.first].move(tc, tc - 1);
      return std::make_pair(0u, 0);
    }
  }

  unsigned customers;
  unsigned tables;
  crp_histogram h[NumFloors];
  template<class Archive> void serialize(Archive& ar, const unsigned int version) {
    ar & customers;
    ar & tables;
    for (unsigned i = 0; i < NumFloors; ++i)
      ar & h[i];
  }
};

template <unsigned N>
void swap(crp_table_manager<N>& a, crp_table_manager<N>& b) {
  std::swap(a.customers, b.customers);
  std::swap(a.tables, b.tables);
  std::swap(a.h, b.h);
}

template <unsigned N>
std::ostream& operator<<(std::ostream& os, const crp_table_manager<N>& tm) {
  os << '[' << tm.num_customers() << " customer" << (tm.num_customers() == 1 ? "" : "s")
     << " at " << tm.num_tables() << " table" << (tm.num_tables() == 1 ? "" : "s") << " |||";
  bool first = true;
  for (unsigned floor = 0; floor < N; ++floor) {
    os << " floor:" << (floor+1) << '/' << N << ' ';
    if (tm.h[floor].begin() == tm.h[floor].end()) { os << "EMPTY"; }
    for (auto& table : tm.h[floor]) {
      if (first) first = false; else os << "  --  ";
      os << '(' << table.first << ") x " << table.second;
    }
  }
  return os << ']';
}

}

#endif
