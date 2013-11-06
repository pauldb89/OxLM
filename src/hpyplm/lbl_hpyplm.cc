#include <iostream>
#include <unordered_map>
#include <cstdlib>

// Boost
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "lbl_hpyplm.h"
#include "corpus/corpus.h"
#include "pyp/m.h"
#include "pyp/random.h"
#include "pyp/crp.h"
#include "pyp/tied_parameter_resampler.h"

#define kORDER 5

static const char *REVISION = "$Rev: 0 $";

using namespace std;
using namespace boost;
using namespace boost::program_options;
using namespace pyp;
using namespace oxlm;


int main(int argc, char** argv) {
  cout << "A product of log-bilinear and PYP language models: Copyright 2013 Phil Blunsom, " 
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
    ("input,i", value<string>()->required(), 
        "corpus of sentences, one per line")
    ("test-set", value<string>(), 
        "corpus of test sentences to be evaluated at each iteration")
    ("iterations", value<int>()->default_value(10), 
        "number of sampling passes through the data")
    ("order,n", value<int>()->default_value(3), 
        "ngram order")
    ("model-in,m", value<string>()->required(),
        "base log-bilinear model")
    ("word-width", value<int>()->default_value(100), 
        "Width of word representation vectors.")
    ("model-out,o", value<string>()->default_value("model"), 
        "filename of model output file")
    ("threads", value<int>()->default_value(1), 
        "number of worker threads.")
    ("verbose,v", "print perplexity for each sentence (1) or input token (2) ")
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

  if (vm.count("help")) { 
    cout << cmdline_options << "\n"; 
    return 1; 
  }
  omp_set_num_threads(vm["threads"].as<int>());

  MT19937 eng;
  string train_file = vm["input"].as<string>();
  int samples = vm["iterations"].as<int>();

  ModelData config;
  config.word_representation_size = vm["word-width"].as<int>();
  config.ngram_order = vm["order"].as<int>();
  Dict dict;
  LogBiLinearModel lbl_model(config, dict);
  {
    std::ifstream f(vm["model-in"].as<string>().c_str());
    boost::archive::text_iarchive ar(f);
    ar >> lbl_model;
  }
  assert (lbl_model.config.ngram_order >= config.ngram_order);
  dict = lbl_model.label_set();
  
  vector<vector<WordId> > corpuse;
  set<WordId> vocabe, tv;
  const WordId kSOS = dict.Convert("<s>");
  const WordId kEOS = dict.Convert("</s>");
  cerr << "Reading corpus...\n";
  ReadFromFile(train_file, &dict, &corpuse, &vocabe);
  cerr << "E-corpus size: " << corpuse.size() << " sentences\t (" << vocabe.size() << " word types)\n";

  vector<vector<WordId> > test;
  if (vm.count("test-set")) {
    string test_file = vm["test-set"].as<string>();
    ReadFromFile(test_file, &dict, &test, &tv);
  }

  LBL_PYPLM<kORDER> lm(lbl_model, 1, 1, 1, 1);
  vector<WordId> ctx(kORDER - 1, kSOS);
  auto count=0;
  for (int sample=0; sample < samples; ++sample) {
    for (const auto& s : corpuse) {
      if (count++ % 100 == 99) cerr << '.' << flush;
      ctx.resize(kORDER - 1);
      for (size_t i = 0; i <= s.size(); ++i) {
        WordId w = (i < s.size() ? s[i] : kEOS);
        if (sample > 0) lm.decrement(w, ctx, eng);
        lm.increment(w, ctx, eng);
        ctx.push_back(w);
      }
    }
    if (sample % 10 == 9) {
      cerr << " [LLH=" << lm.log_likelihood() << "]" << endl;
      if (sample % 30u == 29) lm.resample_hyperparameters(eng);
    } else { cerr << '.' << flush; }
  }
//  lm.print(cerr);
  if (!test.empty()) {
    double llh = 0;
    int cnt = 0;
    int oovs = 0;
    for (auto& s : test) {
      ctx.resize(kORDER - 1);
      for (size_t i = 0; i <= s.size(); ++i) {
        WordId w = (i < s.size() ? s[i] : kEOS);
        double lp = log(lm.prob(w, ctx)) / log(2);
        if (i < s.size() && vocabe.count(w) == 0) {
          //        cerr << "**OOV ";
          ++oovs;
          lp = 0;
        }
        //      cerr << "p(" << dict.Convert(w) << " |";
        //      for (size_t j = ctx.size() + 1 - kORDER; j < ctx.size(); ++j)
        //        cerr << ' ' << dict.Convert(ctx[j]);
        //      cerr << ") = " << lp << endl;
        ctx.push_back(w);
        llh -= lp;
        cnt++;
      }
    }
    cnt -= oovs;
    cerr << "  Log_10 prob: " << (-llh * log(2) / log(10)) << endl;
    cerr << "        Count: " << cnt << endl;
    cerr << "         OOVs: " << oovs << endl;
    cerr << "Cross-entropy: " << (llh / cnt) << endl;
    cerr << "   Perplexity: " << pow(2, llh / cnt) << endl;
  }
  return 0;
}

