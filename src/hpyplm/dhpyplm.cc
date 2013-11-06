#include <iostream>
#include <unordered_map>
#include <cstdlib>

#include "corpus/corpus.h"
#include "pyp/m.h"
#include "pyp/random.h"
#include "pyp/crp.h"
#include "pyp/mf_crp.h"
#include "pyp/tied_parameter_resampler.h"
#include "uvector.h"
#include "dhpyplm.h"

// A not very memory-efficient implementation of a domain adapting
// HPYP language model, as described by Wood & Teh (AISTATS, 2009)
//
// I use templates to handle the recursive formalation of the prior, so
// the order of the model has to be specified here, at compile time:
#define kORDER 3

using namespace std;
using namespace pyp;
using namespace oxlm;

Dict dict;

int main(int argc, char** argv) {
  if (argc < 4) {
    cerr << argv[0] << " <training1.txt> <training2.txt> [...] <test.txt> <nsamples>\n\nInfer a " << kORDER << "-gram HPYP LM and report posterior-predictive perplexity\n100 is usually sufficient for <nsamples>\n";
    return 1;
  }
  MT19937 eng;
  vector<string> train_files;
  for (int i = 1; i < argc - 2; ++i)
    train_files.push_back(argv[i]);
  string test_file = argv[argc - 2];
  int samples = atoi(argv[argc - 1]);
  int d = 1;
  for (auto& tf : train_files)
    cerr << (d++==1 ? "*" : "") << "Corpus "<< ": " << tf << endl;
  set<unsigned> vocab, tvocab;
  const unsigned kSOS = dict.Convert("<s>");
  const unsigned kEOS = dict.Convert("</s>");
  vector<vector<unsigned> > test;
  ReadFromFile(test_file, &dict, &test, &tvocab);
  vector<vector<vector<unsigned> > > corpora(train_files.size());
  d = 0;
  for (const auto& train_file : train_files)
    ReadFromFile(train_file, &dict, &corpora[d++], &vocab);

  vector<vector<unsigned>> corpus = corpora[0];
  cerr << "E-corpus size: " << corpus.size() << " sentences\t (" << vocab.size() << " word types)\n";
  PYPLM<kORDER> latent_lm(vocab.size(), 1, 1, 1, 1);
  vector<DAPYPLM<kORDER>> dlm(corpora.size(), DAPYPLM<kORDER>(latent_lm)); // domain LMs
  vector<unsigned> ctx(kORDER - 1, kSOS);
  for (int sample=0; sample < samples; ++sample) {
    int ci = 0;
    for (const auto& corpus : corpora) {
      DAPYPLM<kORDER>& lm = dlm[ci];
      ++ci;
      for (const auto& s : corpus) {
        ctx.resize(kORDER - 1);
        for (unsigned i = 0; i <= s.size(); ++i) {
          unsigned w = (i < s.size() ? s[i] : kEOS);
          if (sample > 0) lm.decrement(w, ctx, eng);
          lm.increment(w, ctx, eng);
          ctx.push_back(w);
        }
      }
    }
    if (sample % 10 == 9) {
      double llh = latent_lm.log_likelihood();
      for (auto& lm : dlm) llh += lm.log_likelihood();
      cerr << " [LLH=" << llh << "]\n";
      if (sample % 30u == 29) {
        for (auto& lm : dlm) lm.resample_hyperparameters(eng);
        latent_lm.resample_hyperparameters(eng);
      }
    } else { cerr << '.' << flush; }
  }
  double llh = 0;
  unsigned cnt = 0;
  unsigned oovs = 0;
  for (auto& s : test) {
    ctx.resize(kORDER - 1);
    for (unsigned i = 0; i <= s.size(); ++i) {
      unsigned w = (i < s.size() ? s[i] : kEOS);
      double lp = log(dlm[0].prob(w, ctx)) / log(2);
      if (i < s.size() && vocab.count(w) == 0) {
        cerr << "**OOV ";
        ++oovs;
        lp = 0;
      }
      cerr << "p(" << dict.Convert(w) << " |";
      for (unsigned j = ctx.size() + 1 - kORDER; j < ctx.size(); ++j)
        cerr << ' ' << dict.Convert(ctx[j]);
      cerr << ") = " << lp << endl;
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
  return 0;
}

