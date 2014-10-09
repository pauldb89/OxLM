/*
 * @Author: prashant
 * @Date: Sep 10, 2014
 */

#include "lbl/parallel_corpus.h"

#include <boost/make_shared.hpp>

#include "lbl/context_processor.h"
#include "lbl/model_utils.h"
#include "utils/conditional_omp.h"

using namespace std;

namespace oxlm {

ParallelCorpus::ParallelCorpus() {}

ParallelCorpus::ParallelCorpus(
		const string& training_file,
		const string& alignment_file,
		const boost::shared_ptr<Vocabulary>& vocab_base,
		bool convert_unknowns) {
	boost::shared_ptr<ParallelVocabulary> vocab =
	    dynamic_pointer_cast<ParallelVocabulary>(vocab_base);
	assert(vocab != nullptr);

  // If any of the vocabularies has already been set (i.e. contains more than
  // <s> and </s> and therefore has size > 2), make them immutable.
  bool immutable_vocab = vocab->size() > 2;
  bool immutable_source_vocab = vocab->sourceSize() > 2;
	int src_end_id = convertSource(
      "</s>", vocab, immutable_source_vocab, convert_unknowns);
	int end_id = convert("</s>", vocab, immutable_vocab, convert_unknowns);

	ifstream tin(training_file);
	ifstream ain(alignment_file);
  string training_line, alignment_line;
	while (getline(tin, training_line) && getline(ain, alignment_line)) {
    int prev_source_size = srcData.size();
    int prev_target_size = data.size();
    stringstream stream(training_line);
    string token;
    while (stream >> token) {
      if (token == "|||") {
        break;
      }
      int word_id = convertSource(
          token, vocab, immutable_source_vocab, convert_unknowns);
      srcData.push_back(word_id);
    }
    srcData.push_back(src_end_id);

    while (stream >> token) {
      int word_id = convert(token, vocab, immutable_vocab, convert_unknowns);
      data.push_back(word_id);
    }
    data.push_back(end_id);

    alignments.resize(data.size());
    stringstream astream(alignment_line);
    while (astream >> token) {
      size_t pos = token.find("-");
      int src_pos = stoi(token.substr(0, pos));
      int trg_pos = stoi(token.substr(pos + 1));

      int src_index = prev_source_size + src_pos;
      int trg_index = prev_target_size + trg_pos;
      alignments[trg_index].push_back(src_index);
    }

    alignments[data.size() - 1].push_back(srcData.size() - 1);
	}

  // Sort alignment links for each target index.
  for (auto& links: alignments) {
    sort(links.begin(), links.end());
  }
}

ParallelCorpus::ParallelCorpus(
    const vector<int>& source_data,
    const vector<int>& target_data,
    const vector<vector<long long>>& links)
    : Corpus(target_data), srcData(source_data), alignments(links) {}

size_t ParallelCorpus::sourceSize() const {
  return srcData.size();
}

int ParallelCorpus::sourceAt(long long index) const {
  return srcData[index];
}

vector<long long> ParallelCorpus::getLinks(long long index) const {
  return alignments[index];
}

bool ParallelCorpus::isAligned(long long index) const {
  return alignments[index].size() > 0;
}

} // namespace oxlm

BOOST_CLASS_EXPORT_IMPLEMENT(oxlm::ParallelCorpus)
