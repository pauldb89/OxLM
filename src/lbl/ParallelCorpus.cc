/*
 * ParallelCorpus.cc
 *
 *  Created on: Sep 10, 2014
 *      Author: prashant
 */
#include "lbl/ParallelCorpus.h"

#include <boost/make_shared.hpp>

#include "lbl/context_processor.h"
#include "lbl/model_utils.h"
#include "utils/conditional_omp.h"

using namespace std;

namespace oxlm {

ParallelCorpus::ParallelCorpus() {}

ParallelCorpus::~ParallelCorpus() {
	// TODO Auto-generated destructor stub
}

Corpus::Corpus() {}

/*
 * Following is similar to readCorpus
 * here we read parallel aligned corpus
 * build the vocab, align link to target for the source file
 */
Corpus::Corpus(
	    const string& filename, const boost::shared_ptr<Vocabulary>& vocab,
	    bool immutable_dict, bool convert_unknowns) {
	  boost::shared_ptr<MCorpus> corpus = boost::make_shared<MCorpus>();
	  int end_id = convert("</s>", vocab, immutable_dict, convert_unknowns);

	  ifstream in(filename);
	  string line;
	  while (getline(in, line)) {
	    stringstream line_stream(line);
	    string token;
	    while (line_stream >> token) {
	      corpus->push_back(convert(token, vocab, immutable_dict, convert_unknowns));
	    }
	    corpus->push_back(end_id);
	  }
}

ParallelCorpus::ParallelCorpus(
		const string& sourcefile,
		const string& alignfile,
		const boost::shared_ptr<Vocabulary>& vocab_base,
		bool immutable_dict,
		bool convert_unknowns) {
	boost::shared_ptr<ParallelVocabulary> vocab =
		dynamic_pointer_cast<ParallelVocabulary>(vocab_base);
	assert(vocab != nullptr);

	//vector<WordId> m_src_corpus;
	int end_id = convert("</s>", vocab, immutable_dict, convert_unknowns);
	int src_end_id = convertSource("</s>", vocab, immutable_dict, convert_unknowns);
	// Can I make this an auto pointer instead of shared pointer, can it be local to this function?


	ifstream srcin(sourcefile);

	ifstream alignin(alignfile);
	string srcline, trgline, alignline;
	while(srcin.good() && alignin.good()){

		stringstream src_stream;
		src_stream << srcin.rdbuf();

		stringstream align_stream;
		align_stream << alignin.rdbuf();


		string line;
		Sentence src_sent, trg_sent;
		// store the source words in a temporary sentence vector
		while (getline(src_stream, line)) {
		    stringstream line_stream(line);
		    string token;
		    while (line_stream >> token) {
		    	if (token == "|||") {
		    		break;
		    	}
		    	int ID = convertSource(token, vocab, immutable_dict, convert_unknowns);
		    	sourceData.push_back(ID);
		    	src_sent.push_back(ID);
		    }
		    sourceData.push_back(src_end_id);
		    while (line_stream >> token) {
		    	int ID = convert(token, vocab, immutable_dict, convert_unknowns);
		    	data.push_back(ID);
		    	trg_sent.push_back(ID);
		    }
		    data.push_back(end_id);
		}
		string token;
		// store the target words in a temporary sentence vector

		while (align_stream >> token){
			stringstream ss;
			size_t pos = token.find("-");
			string target_pos = token.substr(0, pos);
			string source_pos = token.substr(pos+1);
			// conversion from string to int
			int trg_pos, src_pos;
			ss << target_pos;
			ss >> trg_pos;
			ss << source_pos;
			ss >> src_pos;
			// src_pos and trg_pos are the index (starting from 0) in that sentence pair
			// calculate offset : targetCorpus.size - currentSentence.size + 1
			int trg_offset = data.size() - trg_sent.size() + 1;
			int src_offset = sourceData.size() - src_sent.size() + 1;
			vector<int> src_vec;
			if(parallelData.size() != 0)
				src_vec = parallelData.at(trg_offset+trg_pos);
			src_vec.push_back(src_offset+src_pos);
			parallelData.push_back(src_vec);
		}
		// align eos markers
		vector<int> src_vec;
		src_vec.push_back(sourceData.size()-1);
		parallelData.push_back(src_vec);
		src_vec.empty();
	}

	srcin.close();
	alignin.close();
	m_src_size = sourceData.size();
}

} /* namespace oxlm */
