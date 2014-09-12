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
			bool immutable_dict,
			bool convert_unknowns);

 protected:
  int sourceAt(size_t index) const;

  vector<size_t> getLinks(size_t index) const;

 private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
	  ar & data;
	  ar & srcData;
	  ar & alignments;
	}

 protected:
  vector<int> srcData;
  vector<vector<size_t>> alignments;
};

} // namespace oxlm
