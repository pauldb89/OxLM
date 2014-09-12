#include "lbl/corpus.h"

#include "lbl/model_utils.h"

namespace oxlm {

Corpus::Corpus() {}

/*
 * Reads a monolingual corpus.
 */
Corpus::Corpus(
	    const string& filename, const boost::shared_ptr<Vocabulary>& vocab,
	    bool immutable_dict, bool convert_unknowns) {
  int end_id = convert("</s>", vocab, immutable_dict, convert_unknowns);

  ifstream in(filename);
  string line;
  while (getline(in, line)) {
    stringstream line_stream(line);
    string token;
    while (line_stream >> token) {
      data.push_back(convert(token, vocab, immutable_dict, convert_unknowns));
    }
    data.push_back(end_id);
  }
}

Corpus::Corpus(const vector<int>& data) : data(data) {}

int Corpus::at(int index) const {
  return data[index];
}

size_t Corpus::size() const {
  return data.size();
}

Corpus::~Corpus() {}

} // namespace oxlm
