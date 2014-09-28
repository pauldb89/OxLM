#pragma once

#include <boost/make_shared.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include "lbl/corpus.h"
#include "lbl/vocabulary.h"

using namespace std;

namespace oxlm {

class ParallelCorpus : public Corpus {
 public:
  ParallelCorpus();

  ParallelCorpus(
  		const string& training_file,
		  const string& alignment_file,
			const boost::shared_ptr<Vocabulary>& vocab,
			bool convert_unknowns);

  ParallelCorpus(
      const vector<int>& source_data,
      const vector<int>& target_data,
      const vector<vector<long long>>& links);

  size_t sourceSize() const;

  int sourceAt(long long index) const;

  vector<long long> getLinks(long long index) const;

  bool isAligned(long long index) const;

 private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
    ar & boost::serialization::base_object<Corpus>(*this);

	  ar & srcData;
	  ar & alignments;
	}

 protected:
  vector<int> srcData;
  vector<vector<long long>> alignments;
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::ParallelCorpus)
