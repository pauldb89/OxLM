#pragma once

#include "lbl/archive_export.h"
#include "lbl/vocabulary.h"

namespace oxlm {

class Corpus {
 public:
  Corpus();

	Corpus(
      const string& filename,
      const boost::shared_ptr<Vocabulary>& vocab,
      bool convert_unknowns);

	Corpus(const vector<int>& data);

	int at(int index) const;

  size_t size() const;

  virtual ~Corpus();

 private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & data;
	}

 protected:
	vector<int> data;
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::Corpus)
