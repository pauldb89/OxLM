#include <iostream>
#include <unordered_map>
#include <cstdlib>

#include "chpyplm.h"
#include "corpus/corpus.h"
#include "pyp/m.h"
#include "pyp/random.h"
#include "pyp/crp.h"
#include "pyp/tied_parameter_resampler.h"

#define kWORDER 5
#define kCORDER 6

using namespace std;
using namespace pyp;
using namespace oxlm;

Dict dict;

int main(int argc, char** argv) {
  if (argc != 4) {
    cerr << argv[0] << " <training.txt> <ntraining_passes> <nsamples> \n\nEstimate a " 
         << '(' << kWORDER << ',' << kCORDER << ")-gram CHPYP LM and then generates nsamples from it.\n";
    return 1;
  }
  MT19937 eng;
  string train_file = argv[1];
  int training_samples = atoi(argv[2]);
  int samples = atoi(argv[3]);
  
  vector<vector<unsigned> > corpuse;
  set<unsigned> vocabe, tv;
  const unsigned kSOS = dict.Convert("<s>");
  const unsigned kEOS = dict.Convert("</s>");
  cerr << "Reading corpus...\n";
  ReadFromFile(train_file, &dict, &corpuse, &vocabe);

  // index the individual characters of the tokens
  Dict char_dict;
  char_dict.Convert("<s>");
  char_dict.Convert("/");
  char_dict.Convert("<");
  char_dict.Convert(">");
  char_dict.Convert(" ");
  for (const auto& s : corpuse)
    for (const auto& e : s) {
      const std::string w = dict.Convert(e);
      for (size_t i=0; i<w.size(); ++i)
        char_dict.Convert(w.substr(i,1));
    }

  cerr << "E-corpus size: " << corpuse.size() << " sentences\t (" << vocabe.size() 
       << " word types, " << char_dict.max() << " characters)\n";
  
  CHPYPLM<kWORDER,kCORDER> lm(char_dict, 1, 1, 1, 1);
  vector<unsigned> ctx(kWORDER - 1, kSOS);
  for (int sample=0; sample < training_samples; ++sample) {
    for (const auto& s : corpuse) {
      ctx.resize(kWORDER - 1);
      for (unsigned i = 0; i <= s.size(); ++i) {
        unsigned w = (i < s.size() ? s[i] : kEOS);
        if (sample > 0) lm.decrement(dict.Convert(w), ctx, eng);
        lm.increment(dict.Convert(w), ctx, eng);
        ctx.push_back(w);
      }
    }
    if (sample % 10 == 9) {
      cerr << " [LLH=" << lm.log_likelihood() << "]" << endl;
      if (sample % 30u == 29) lm.resample_hyperparameters(eng);
    } else { cerr << '.' << flush; }
  }
  cerr << endl;
  for (int sample=0; sample < samples; ++sample) {
    std::string w = lm.generate(ctx, eng);
    unsigned w_i = dict.Convert(w);
    if (w_i == kEOS) {
      cout << endl;
      ctx.resize(kWORDER - 1);
    }
    else 
      cout << w << " ";
    ctx.push_back(w_i);
  }

  return 0;
}

