#include "lbl/model_utils.h"

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/make_shared.hpp>

#include "lbl/context_processor.h"
#include "utils/conditional_omp.h"

namespace oxlm {

vector<int> scatterMinibatch(const vector<int>& minibatch) {
  size_t thread_id = omp_get_thread_num();
  size_t num_threads = omp_get_num_threads();

  vector<int> result;
  result.reserve(minibatch.size() / num_threads + 1);
  for (size_t s = thread_id; s < minibatch.size(); s += num_threads) {
    result.push_back(minibatch.at(s));
  }

  return result;
}

void loadClassesFromFile(
    const string& class_file, const string& training_file,
    vector<int>& classes, const boost::shared_ptr<Vocabulary>& vocab,
    VectorReal& class_bias) {
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
    int w_id = vocab->convert(token_str);

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
  classes.push_back(vocab->size());

  class_bias = VectorReal::Zero(class_freqs.size());
  for (size_t i = 0; i < class_freqs.size(); ++i) {
    class_bias(i) = log(class_freqs.at(i)) - log(total_mass);
  }

  cout << "Read " << vocab->size() << " types in "
       << classes.size() - 1 << " classes with an average of "
       << vocab->size() / float(classes.size() - 1) << " types per bin." << endl;

  in.close();
}

void frequencyBinning(
    const string& training_file, int num_classes,
    vector<int>& classes, const boost::shared_ptr<Vocabulary>& vocab,
    VectorReal& class_bias) {
  ifstream in(training_file);
  string line, token;

  map<string, int> tmp_dict;
  vector<pair<string, int>> counts;
  int num_tokens = 0, num_eos_tokens = 0;
  string eos = "</s>";

  while (getline(in, line)) {
    stringstream line_stream(line);
    while (line_stream >> token) {
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
    WordId id = vocab->convert(counts.at(i).first);
    mass += counts.at(i).second;

    if (mass > bin_size) {
      remaining_tokens -= mass;
      bin_size = remaining_tokens / (num_classes - classes.size());
      class_bias(classes.size() - 1) = log(mass);
      classes.push_back(id + 1);
      mass = 0;
    }
  }

  if (classes.back() != int(vocab->size())) {
    class_bias(classes.size() - 1) = log(mass);
    classes.push_back(vocab->size());
  }


  assert(classes.size() == num_classes + 1);
  class_bias.array() -= log(num_tokens + num_eos_tokens);

  cout << "Binned " << vocab->size() << " types in "
       << classes.size() - 1 << " classes with an average of "
       << vocab->size() / float(classes.size() - 1) << " types per bin." << endl;
  in.close();
}

int convert(
    const string& token, const boost::shared_ptr<Vocabulary>& vocab,
    bool immutable_vocab, bool convert_unknowns) {
  int word_id = vocab->convert(token, immutable_vocab);
  if (word_id < 0) {
    if (convert_unknowns) {
      word_id = vocab->convert("<unk>", immutable_vocab);
      assert(word_id >= 0);
    } else {
      cout << token << " " << word_id << endl;
      assert(!"Unknown word found in test corpus.");
    }
  }

  return word_id;
}

int convertSource(
    const string& token,
    const boost::shared_ptr<ParallelVocabulary>& vocab,
    bool immutable_vocab,
    bool convert_unknowns) {
  int word_id = vocab->convertSource(token, immutable_vocab);
  if (word_id < 0) {
    if (convert_unknowns) {
      word_id = vocab->convertSource("<unk>", immutable_vocab);
      assert(word_id >= 0);
    } else {
      cout << token << " " << word_id << endl;
      assert(!"Unknown source word found in test corpus.");
    }
  }

  return word_id;
}

boost::shared_ptr<Corpus> readTrainingCorpus(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<Vocabulary>& vocab,
    bool convert_unknowns) {
  cerr << "Reading training corpus..." << endl;
  boost::shared_ptr<Corpus> corpus;
  if (config->source_order == 0) {
	  corpus = boost::make_shared<Corpus>(
        config->training_file, vocab, convert_unknowns);
    config->vocab_size = vocab->size();
  } else {
    boost::shared_ptr<ParallelVocabulary> parallel_vocab =
        dynamic_pointer_cast<ParallelVocabulary>(vocab);
    assert(parallel_vocab != nullptr);

	  corpus = boost::make_shared<ParallelCorpus>(
        config->training_file, config->alignment_file, parallel_vocab,
        convert_unknowns);

    config->vocab_size = parallel_vocab->size();
    config->source_vocab_size = parallel_vocab->sourceSize();
  }
  cerr << "Done reading training corpus..." << endl;

  return corpus;
}

boost::shared_ptr<Corpus> readTestCorpus(
    const boost::shared_ptr<ModelData>& config,
    const boost::shared_ptr<Vocabulary>& vocab,
    bool convert_unknowns) {
  cerr << "Reading test corpus..." << endl;
  boost::shared_ptr<Corpus> corpus;
  if (config->source_order == 0) {
	  corpus = boost::make_shared<Corpus>(
        config->test_file, vocab, convert_unknowns);
  } else {
	  corpus = boost::make_shared<ParallelCorpus>(
        config->test_file, config->test_alignment_file, vocab,
        convert_unknowns);
  }
  cerr << "Done reading test corpus..." << endl;

  return corpus;
}

Real perplexity(Real log_likelihood, size_t corpus_size) {
  return exp(log_likelihood / corpus_size);
}

} // namespace oxlm
