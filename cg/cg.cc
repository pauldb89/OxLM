// STL
#include <vector>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <random>

// Boost
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/lexical_cast.hpp>

// Eigen
#include <Eigen/Core>

// Local
#include "cg/cnlm.h"
#include "corpus/corpus.h"

static const char *REVISION = "$Rev: 247 $";

// Namespaces
using namespace boost;
using namespace boost::program_options;
using namespace std;
using namespace oxlm;
using namespace Eigen;


Real perplexity(const ConditionalNLM& model, const vector<Sentence>& test_source_corpus, const vector<Sentence>& test_target_corpus);


int main(int argc, char **argv) {
  cerr << "Conditional generation from neural translation models: Copyright 2013 Phil Blunsom, " 
       << REVISION << '\n' << endl;

  ///////////////////////////////////////////////////////////////////////////////////////
  // Command line processing
  variables_map vm; 

  // Command line processing
  options_description cmdline_specific("Command line specific options");
  cmdline_specific.add_options()
    ("help,h", "print help message")
    ("config,c", value<string>(), 
        "config file specifying additional command line options")
    ;
  options_description generic("Allowed options");
  generic.add_options()
    ("source,s", value<string>(), 
        "corpus of sentences, one per line")
    ("samples", value<int>()->default_value(1), 
        "number of samples from the nlm")
    ("k-best", value<int>()->default_value(1), 
        "output k-best samples.")
    ("model-in,m", value<string>(), 
        "model to generate from")
    ("threads", value<int>()->default_value(1), 
        "number of worker threads.")
    ("word-penalty", value<double>()->default_value(0), 
        "word penalty added to the sample log prob for each word generated.")
    ;
  options_description config_options, cmdline_options;
  config_options.add(generic);
  cmdline_options.add(generic).add(cmdline_specific);

  store(parse_command_line(argc, argv, cmdline_options), vm); 
  if (vm.count("config") > 0) {
    ifstream config(vm["config"].as<string>().c_str());
    store(parse_config_file(config, cmdline_options), vm); 
  }
  notify(vm);
  ///////////////////////////////////////////////////////////////////////////////////////

  if (vm.count("help") || !vm.count("source") || !vm.count("model-in")) { 
    cerr << cmdline_options << "\n"; 
    return 1; 
  }

  omp_set_num_threads(vm["threads"].as<int>());

  ConditionalNLM model;
  std::ifstream f(vm["model-in"].as<string>().c_str());
  boost::archive::text_iarchive ar(f);
  ar >> model;
  cerr << model.length_ratio << endl;

  WordId start_id = model.label_set().Lookup("<s>");
  WordId end_id = model.label_set().Lookup("</s>");

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Real> distribution(0.0,1.0);
  
  //////////////////////////////////////////////
  // process the input sentences
  string line, token;
  ifstream source_in(vm["source"].as<string>().c_str());
  int source_counter=0;
  while (getline(source_in, line)) {
    // read the sentence
    stringstream line_stream(line);
    Sentence s;
    while (line_stream >> token) 
      s.push_back(model.source_label_set().Convert(token));
    s.push_back(end_id);

    // sample a translation
    //map<int, set< pair<Real,Sentence> > > samples;
    set< pair<Real,Sentence> > samples;
    VectorReal prediction_vector, source_vector, class_probs, word_probs;

    for (int sample=0; sample < vm["samples"].as<int>(); ++sample) {
      Sentence sample(model.config.ngram_order-1, start_id);
      Real sample_log_prob=0;
      while (sample.back() != end_id) {
        vector<WordId> context(sample.end()-model.config.ngram_order+1, sample.end());
        Real point = distribution(gen);
        Real accumulator=0.0;

        model.source_representation(s, sample.size()-model.config.ngram_order+1, source_vector);
        model.hidden_layer(context, source_vector, prediction_vector);

        // sample a class
        model.class_log_probs(context, source_vector, prediction_vector, class_probs, false);

//        class_probs.array() *= 0.7;
//        class_probs.array() -= log(class_probs.array().exp().sum());

        int c=0;
        Real clp=0;
        for (; c < class_probs.rows(); ++c) {
          clp = class_probs(c);
          if ((accumulator += exp(clp)) >= point || c == class_probs.rows()-1) break;
        }

        // sample a word
        point = distribution(gen);
        accumulator=0;
        model.word_log_probs(c, context, source_vector, prediction_vector, word_probs, false);

//        word_probs.array() *= 0.7; 
//        word_probs.array() -= log(word_probs.array().exp().sum());

        for (WordId w=0; w < word_probs.rows(); ++w) {
          Real wlp = word_probs(w);
          if ((accumulator += exp(wlp)) >= point || w == word_probs.rows()-1) {
            sample.push_back(model.map_class_to_word_index(c, w));
            sample_log_prob += clp+wlp + vm["word-penalty"].as<double>();
//            sample_log_prob += clp+wlp;
            break;
          }
        }
      }
//      sample_log_prob -= (vm["word-penalty"].as<double>() * floor(fabs(Real(s.size()) - (model.length_ratio*(sample.size() - model.config.ngram_order)))));
      
//      samples[sample.size()-model.config.ngram_order].insert(
//              make_pair(-sample_log_prob, Sentence(sample.begin()+model.config.ngram_order-1, sample.end()-1)));
      samples.insert(make_pair(-sample_log_prob, Sentence(sample.begin()+model.config.ngram_order-1, sample.end()-1)));
    }
    /*
    for (auto s : samples) {
      int c=0;
      for (auto p : s.second) {
        cout << source_counter << " ||| " << s.first << " ||| ";
        for (auto w : p.second)
          cout << model.label_set().Convert(w) << " ";
        cout << "||| " << p.first << endl;
        if (++c > 5) break;
      }
    }
    */
    if (vm["k-best"].as<int>() == 1) {
      for (auto w : samples.begin()->second)
        cout << model.label_set().Convert(w) << " ";
      cout << endl;
    }
    else {
      int c=0;
      for (auto s : samples) {
        cout << source_counter << " ||| ";
        for (auto w : s.second)
          cout << model.label_set().Convert(w) << " ";
        cout << "||| " << s.first << endl;
        if (++c >= vm["k-best"].as<int>()) break;
      }
    }
    source_counter++;
  }
  source_in.close();
  //////////////////////////////////////////////

  return 0;
}

Real perplexity(const ConditionalNLM& model, const vector<Sentence>& test_source_corpus, const vector<Sentence>& test_target_corpus) {
  Real p=0.0;

  int context_width = model.config.ngram_order-1;
  int tokens=0;
  WordId start_id = model.label_set().Lookup("<s>");

  #pragma omp master
  cerr << "Calculating perplexity for " << test_target_corpus.size() << " sentences";

  std::vector<WordId> context(context_width);

  size_t thread_num = omp_get_thread_num();
  size_t num_threads = omp_get_num_threads();
  for (size_t s = thread_num; s < test_target_corpus.size(); s += num_threads) {
    const Sentence& sent = test_target_corpus.at(s);
    for (size_t t_i=0; t_i < sent.size(); ++t_i) {
      WordId w = sent.at(t_i);
      int context_start = t_i - context_width;
      bool sentence_start = (t_i==0);

      for (int i=context_width-1; i>=0; --i) {
        int j=context_start+i;
        sentence_start = (sentence_start || j<0);
        int v_i = (sentence_start ? start_id : sent.at(j));

        context.at(i) = v_i;
      }
      Real log_prob = model.log_prob(w, context, test_source_corpus.at(s), true, t_i);
      p += log_prob;

      #pragma omp master
      if (tokens % 1000 == 0) { cerr << "."; cerr.flush(); }

      tokens++;
    }
  }
  #pragma omp master
  cerr << endl;

  return p;
}

