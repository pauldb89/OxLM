#include "lbl/model_utils.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/make_shared.hpp>

#include "lbl/context_processor.h"
#include "utils/conditional_omp.h"

namespace oxlm {

vector<int> scatterMinibatch(int start, int end, const vector<int>& indices) {
  size_t thread_num = omp_get_thread_num();
  size_t num_threads = omp_get_num_threads();

  vector<int> result;
  result.reserve((end - start) / num_threads);
  for (int s = start + thread_num; s < end; s += num_threads) {
    result.push_back(indices.at(s));
  }

  return result;
}

void loadClassesFromFile(
    const string& class_file, const string& training_file,
    vector<int>& classes, Dict& dict, VectorReal& class_bias) {
  ifstream tin(training_file);
  string line;
  int num_eos_tokens = 0;
  while (getline(tin, line)) {
    ++num_eos_tokens;
  }

  vector<int> class_freqs(1, num_eos_tokens);
  classes.clear();
  classes.push_back(0);
  classes.push_back(2);

  int mass = 0, total_mass = num_eos_tokens;
  ifstream in(class_file);
  string prev_class_str, class_str, token_str, freq_str;
  while (in >> class_str >> token_str >> freq_str) {
    int w_id = dict.Convert(token_str);

    if (!prev_class_str.empty() && class_str != prev_class_str) {
      class_freqs.push_back(mass);
      classes.push_back(w_id);
      mass = 0;
    }

    int freq = boost::lexical_cast<int>(freq_str);
    mass += freq;
    total_mass += freq;

    prev_class_str = class_str;
  }

  class_freqs.push_back(mass);
  classes.push_back(dict.size());

  class_bias = VectorReal::Zero(class_freqs.size());
  for (size_t i = 0; i < class_freqs.size(); ++i) {
    class_bias(i) = log(class_freqs.at(i)) - log(total_mass);
  }

  cout << "Read " << dict.size() << " types in "
       << classes.size() - 1 << " classes with an average of "
       << dict.size() / float(classes.size() - 1) << " types per bin." << endl;

  in.close();
}

void frequencyBinning(
    const string& training_file, int num_classes,
    vector<int>& classes, Dict& dict, VectorReal& class_bias) {
  ifstream in(training_file);
  string line, token;

  map<string, int> tmp_dict;
  vector<pair<string, int>> counts;
  int num_tokens = 0, num_eos_tokens = 0;
  string eos = "</s>";

  while (getline(in, line)) {
    stringstream line_stream(line);
    while (line_stream >> token) {
      if (token == eos) continue;
      int w_id = tmp_dict.insert(make_pair(token, tmp_dict.size())).first->second;
      assert (w_id <= int(counts.size()));
      if (w_id == int(counts.size())) {
        counts.push_back(make_pair(token, 1));
      } else {
        counts[w_id].second += 1;
      }
      ++num_tokens;
    }
    ++num_eos_tokens;
  }

  sort(counts.begin(), counts.end(),
       [](const pair<string, int>& a, const pair<string, int>& b) -> bool {
           return a.second > b.second;
       });

  classes.clear();
  classes.push_back(0);
  classes.push_back(2);

  class_bias = VectorReal::Zero(num_classes);
  class_bias(0) = log(num_eos_tokens);

  int remaining_tokens = num_tokens;
  int bin_size = remaining_tokens / (num_classes - 1);
  int mass = 0;
  for (size_t i = 0; i < counts.size(); ++i) {
    WordId id = dict.Convert(counts.at(i).first);
    mass += counts.at(i).second;

    if (mass > bin_size) {
      remaining_tokens -= mass;
      bin_size = remaining_tokens / (num_classes - classes.size());
      class_bias(classes.size() - 1) = log(mass);
      classes.push_back(id + 1);
      mass=0;
    }
  }

  if (classes.back() != int(dict.size())) {
    classes.push_back(dict.size());
  }

  assert(classes.size() == num_classes + 1);
  class_bias.array() -= log(num_eos_tokens + num_tokens);

  cout << "Binned " << dict.size() << " types in "
       << classes.size() - 1 << " classes with an average of "
       << dict.size() / float(classes.size() - 1) << " types per bin." << endl;
  in.close();
}

int convert(
    const string& token, Dict& dict,
    bool immutable_dict, bool convert_unknowns) {
  int w = dict.Convert(token, immutable_dict);
  if (w < 0) {
    if (convert_unknowns) {
      w = dict.Convert("<unk>", immutable_dict);
      assert(w >= 0);
    } else {
      cout << token << " " << w << endl;
      assert(!"Unknown word found in test corpus.");
    }
  }
  return w;
}

boost::shared_ptr<Corpus> readCorpus(
    const string& filename, Dict& dict,
    bool immutable_dict, bool convert_unknowns) {
  boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>();
  int end_id = convert("</s>", dict, immutable_dict, convert_unknowns);

  ifstream in(filename);
  string line;
  while (getline(in, line)) {
    stringstream line_stream(line);
    string token;
    while (line_stream >> token) {
      corpus->push_back(convert(token, dict, immutable_dict, convert_unknowns));
    }
    corpus->push_back(end_id);
  }

  return corpus;
}

Real perplexity(Real log_likelihood, size_t corpus_size) {
  return exp(log_likelihood / corpus_size);
}

} // namespace oxlm
