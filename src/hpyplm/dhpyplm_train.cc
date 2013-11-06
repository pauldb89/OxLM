#include <iostream>
#include <unordered_map>
#include <cstdlib>

#include "corpus/corpus.h"
#include "cpyp/m.h"
#include "cpyp/random.h"
#include "cpyp/crp.h"
#include "cpyp/mf_crp.h"
#include "cpyp/tied_parameter_resampler.h"
#include "uvector.h"
#include "dhpyplm.h"

#include "cpyp/boost_serializers.h"
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_oarchive.hpp>

// A not very memory-efficient implementation of a domain adapting
// HPYP language model, as described by Wood & Teh (AISTATS, 2009)
//
// I use templates to handle the recursive formalation of the prior, so
// the order of the model has to be specified here, at compile time:
#define kORDER 3

using namespace std;
using namespace cpyp;

Dict dict;

int main(int argc, char** argv) {
  if (argc < 4) {
    cerr << argv[0] << " <training1.txt> <training2.txt> [...] <output.dlm> <nsamples>\n\nInfer a " << kORDER << "-gram HPYP LM and write the trained model\n100 is usually sufficient for <nsamples>\n";
    return 1;
  }
  MT19937 eng;
  vector<string> train_files;
  for (int i = 1; i < argc - 2; ++i)
    train_files.push_back(argv[i]);
  string output_file = argv[argc - 2];
  int samples = atoi(argv[argc - 1]);
  assert(samples > 0);
  {
    ifstream test(output_file);
    if (test.good()) {
      cerr << "File " << output_file << " appears to exist: please remove\n";
      return 1;
    }
  }

  int d = 1;
  for (auto& tf : train_files)
    cerr << (d++==1 ? "  [primary] " : "[secondary] ")
         << "training corpus "<< ": " << tf << endl;
  set<unsigned> vocab;
  const unsigned kSOS = dict.Convert("<s>");
  const unsigned kEOS = dict.Convert("</s>");
  vector<vector<vector<unsigned> > > corpora(train_files.size());
  d = 0;
  for (const auto& train_file : train_files)
    ReadFromFile(train_file, &dict, &corpora[d++], &vocab);

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
  cerr << "Writing LM to " << output_file << " ...\n";
  ofstream ofile(output_file.c_str(), ios::out | ios::binary);
  if (!ofile.good()) {
    cerr << "Failed to open " << output_file << " for writing\n";
    return 1;
  }
  boost::archive::binary_oarchive oa(ofile);
  oa & dict;
  oa & latent_lm;
  unsigned num_domains = dlm.size();
  oa & num_domains;
  for (unsigned i = 0; i < num_domains; ++i)
    oa & dlm[i];
  return 0;
}

