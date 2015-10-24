#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>

#include "lbl/context_processor.h"
#include "lbl/corpus.h"
#include "lbl/feature_context_generator.h"
#include "lbl/hashed_ngram.h"
#include "lbl/model_utils.h"
#include "lbl/vocabulary.h"
#include "lbl/word_to_class_index.h"

using namespace boost::program_options;
using namespace oxlm;
using namespace std;


void PartitionNGramHashes(
    const string& training_file, const string& class_file,
    const string& tmp_prefix, int feature_context_size, int num_buckets) {
  cout << "Splitting ngrams into " << num_buckets << " files..." << endl;

  auto start_time = GetTime();
  cout << "Reading input data..." << endl;
  boost::shared_ptr<Vocabulary> vocab = boost::make_shared<Vocabulary>();
  vector<int> classes;
  VectorReal classBias;
  loadClassesFromFile(class_file, training_file, classes, vocab, classBias);
  boost::shared_ptr<WordToClassIndex> index = boost::make_shared<WordToClassIndex>(classes);

  boost::shared_ptr<Corpus> corpus = boost::make_shared<Corpus>(training_file, vocab, false);
  boost::shared_ptr<ContextProcessor> processor =
    boost::make_shared<ContextProcessor>(corpus, feature_context_size);
  boost::shared_ptr<FeatureContextGenerator> generator =
    boost::make_shared<FeatureContextGenerator>(feature_context_size);

  cout << "Reading input data took " << GetDuration(start_time, GetTime())
       << " seconds" << endl;

  start_time = GetTime();
  cout << "Iterating over n-grams..." << endl;
  vector<boost::shared_ptr<ofstream>> output_streams;
  for (int i = 0; i < num_buckets; ++i) {
    string filename = tmp_prefix + to_string(i);
    output_streams.push_back(boost::make_shared<ofstream>(filename));
  }

  size_t num_ngrams = 0;
  vector<size_t> ngrams_per_bucket(num_buckets);
  hash<HashedNGram> hasher;
  for (size_t i = 0; i < corpus->size(); ++i) {
    int word_id = corpus->at(i);
    int class_id = index->getClass(word_id);

    vector<WordId> context = processor->extract(i);
    vector<Hash> context_hashes = generator->getFeatureContextHashes(context);

    for (Hash context_hash: context_hashes) {
      HashedNGram ngram(word_id, class_id, context_hash);
      size_t ngram_hash = hasher(ngram);

      ++num_ngrams;
      int bucket = ngram_hash % num_buckets;
      ++ngrams_per_bucket[bucket];
      (*output_streams[bucket]) << ngram_hash << "\n";
    }
  }

  cout << "Total n-grams processed: " << num_ngrams << endl;
  for (int i = 0; i < num_buckets; ++i) {
    cout << ngrams_per_bucket[i] << " n-gram hashes written to bucket " << i << endl;
    output_streams[i]->close();
  }
  cout << "Iterating and splitting n-grams took "
       << GetDuration(start_time, GetTime()) << " seconds.." << endl;
}

void CountAndSortNGramHashes(const string& tmp_prefix, int num_buckets) {
  auto start_time = GetTime();
  cout << "Counting and sorting hashes..." << endl;

  size_t distinct_hashes = 0;
  for (int i = 0; i < num_buckets; ++i) {
    auto bucket_start_time = GetTime();
    cout << "Processing file " << i << "..." << endl;
    string filename = tmp_prefix + to_string(i);
    ifstream fin(filename);

    size_t num_ngrams = 0;
    size_t ngram_hash;
    unordered_map<Hash, int> ngram_counts;
    while (fin >> ngram_hash) {
      ++num_ngrams;
      ++ngram_counts[ngram_hash];
    }
    fin.close();
    cout << "Bucket " << i << " has " << num_ngrams << " total hashes and "
         << ngram_counts.size() << " distinct hashes" << endl;

    distinct_hashes += ngram_counts.size();
    vector<pair<Hash, int>> ngram_hashes;
    for (const auto& entry: ngram_counts) {
      ngram_hashes.push_back(entry);
    }

    sort(
        ngram_hashes.begin(),
        ngram_hashes.end(),
        [](const pair<Hash, int>& lhs, const pair<Hash, int>& rhs) -> bool {
          return lhs.second > rhs.second;
        });

    ofstream fout(filename);
    for (const auto& entry: ngram_hashes) {
      fout << entry.first << " " << entry.second << "\n";
    }
    fout.close();
    cout << "Processing file " << i << " took " << GetDuration(bucket_start_time, GetTime())
         << " seconds..." << endl;
  }
  cout << "Total distinct hashes: " << distinct_hashes << endl;
  cout << "Counting and sorting hashes took " << GetDuration(start_time, GetTime())
       << " seconds..." << endl;
}

void ExtractTopNGramHashes(
    const string& output_file, const string& tmp_prefix, int num_buckets,
    int max_ngrams, int min_ngram_freq) {
  auto start_time = GetTime();
  cout << "Extracting top n-gram hashes..." << endl;
  set<pair<int, pair<Hash, int>>, greater<pair<int, pair<Hash, int>>>> candidates;
  vector<boost::shared_ptr<ifstream>> input_streams;
  for (int i = 0; i < num_buckets; ++i) {
    string filename = tmp_prefix + to_string(i);
    input_streams.push_back(boost::make_shared<ifstream>(filename));

    Hash ngram_hash;
    int ngram_count;
    if ((*input_streams[i]) >> ngram_hash >> ngram_count) {
      candidates.insert(make_pair(ngram_count, make_pair(ngram_hash, i)));
    }
  }

  ofstream fout(output_file);
  int num_ngrams = 0, ngram_freq = 0;
  while (candidates.size()) {
    if (max_ngrams > 0 && num_ngrams > max_ngrams) {
      break;
    }

    ngram_freq = candidates.begin()->first;
    if (ngram_freq < min_ngram_freq) {
      break;
    }

    auto top_candidate = candidates.begin()->second;
    candidates.erase(candidates.begin());
    fout << top_candidate.first << "\n";

    int bucket = top_candidate.second;
    Hash ngram_hash;
    int ngram_count;
    if ((*input_streams[bucket]) >> ngram_hash >> ngram_count) {
      candidates.insert(make_pair(ngram_count, make_pair(ngram_hash, bucket)));
    }

    ++num_ngrams;
  }

  fout.close();
  for (int i = 0; i < num_buckets; ++i) {
    input_streams[i]->close();
  }

  cout << num_ngrams << " were selected" << endl;
  cout << "The minimum frequency is " << max(min_ngram_freq, ngram_freq) << endl;
  cout << "Extracting top n-grams took " << GetDuration(start_time, GetTime())
       << " seconds..." << endl;
}

int main(int argc, char** argv) {
  options_description cmdline_specific("Command line specific options");
  cmdline_specific.add_options()
    ("help,h", "Print help message.")
    ("config,c", value<string>(), "Path to config file.");

  options_description config_options("Allowed options");
  config_options.add_options()
    ("training-file,t", value<string>()->required(), "File containing training data.")
    ("class-file", value<string>()->required(), "File containing word to class mappings.")
    ("ngram-file", value<string>()->required(), "Output file with most frequent n-grams.")
    ("tmp-prefix", value<string>()->required(), "Temporary file prefixes.")
    ("num-buckets", value<int>()->default_value(100), "Number of buckets")
    ("feature-context-size", value<int>()->default_value(4), "Size of the window for n-grams")
    ("max-ngrams", value<int>()->default_value(0), "Maximum n-gram hashes to return.")
    ("min-ngram-freq", value<int>()->default_value(1), "Return all n-gram hashes with frequency above this value.");

  options_description cmdline_options;
  cmdline_options.add(cmdline_specific).add(config_options);

  variables_map vm;
  store(parse_command_line(argc, argv, cmdline_options), vm);
  if (vm.count("help")) {
    cout << cmdline_options << endl;
    return 0;
  }

  if (vm.count("config")) {
    ifstream config(vm["config"].as<string>().c_str());
    store(parse_config_file(config, config_options), vm);
  }

  notify(vm);

  string training_file = vm["training-file"].as<string>();
  string class_file = vm["class-file"].as<string>();
  string output_file = vm["ngram-file"].as<string>();
  string tmp_prefix = vm["tmp-prefix"].as<string>();
  int num_buckets = vm["num-buckets"].as<int>();

  int max_ngrams = vm["max-ngrams"].as<int>();
  int min_ngram_freq = vm["min-ngram-freq"].as<int>();
  int feature_context_size = vm["feature-context-size"].as<int>();

  if (max_ngrams <= 0 && min_ngram_freq <= 1) {
    cout << "You must set either max-ngrams or min-ngrams-freq." << endl;
  }

  PartitionNGramHashes(training_file, class_file, tmp_prefix, feature_context_size, num_buckets);

  CountAndSortNGramHashes(tmp_prefix, num_buckets);

  ExtractTopNGramHashes(output_file, tmp_prefix, num_buckets, max_ngrams, min_ngram_freq);

  return 0;
}
