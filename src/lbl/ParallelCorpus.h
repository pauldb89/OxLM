#pragma once

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/make_shared.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include "lbl/vocabulary.h"

//#include <lbl/model_utils.h>

using namespace std;

namespace oxlm {

typedef int WordId;
typedef std::vector<WordId> Sentence;
typedef std::vector< std::vector<WordId> > PCorpus;
typedef std::vector<WordId> MCorpus;
typedef int Position;

class Corpus {
public:
   Corpus();
	Corpus(const string& filename, const boost::shared_ptr<Vocabulary>& vocab,
		    bool immutable_dict, bool convert_unknowns);
	Corpus(vector<WordId> &data) { this->data = data; }
	WordId at(Position position) { return data[position]; }
	size_t size() { return data.size(); }
	WordId operator[](Position position) { return at(position); }
	void push_back(WordId word_id) { data.push_back(word_id); }
	virtual int identify_me() { return 1; } //dummy virtual function to let dynamic cast work

protected:
	vector<WordId> data;
private:
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & data;
	}
};


class ParallelCorpus : public Corpus {

	friend class boost::serialization::access;
	template<class Archive>
	  void serialize(Archive& ar, const unsigned int version) {
	    ar & data;
	    ar & sourceData;
	    ar & parallelData;
	  }

	size_t m_src_size;

protected:
 vector<WordId> sourceData;
 vector< vector<int> > parallelData;


public:
   ParallelCorpus();
	ParallelCorpus(const string& sourcefile,
			const string& alignfile,
			const boost::shared_ptr<Vocabulary>& vocab,
			bool immutable_dict,
			bool convert_unknowns);
	virtual ~ParallelCorpus();

	Position sourceSize() { return sourceData.size(); }
	//inherited WordId at(Position pos)
	WordId sourceAt(Position position) { return sourceData[position]; }
	vector<Position> &getAlignment(Position position) { return parallelData[position]; }

	void setSource(vector<WordId> &src) { sourceData = src; }
	void setAlignment(vector<vector<Position> > &alignment) { parallelData = alignment; }

};

} /* namespace oxlm */

