#include <iostream>
#include <unordered_map>
#include <cstdlib>

#include "hpyplm.h"
#include "corpus/corpus.h"
#include "pyp/m.h"
#include "pyp/random.h"
#include "pyp/crp.h"
#include "pyp/tied_parameter_resampler.h"

#include "cpyp/boost_serializers.h"
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_oarchive.hpp>

#define kORDER 4

using namespace std;
using namespace oxlm;

Dict dict;

int main(int argc, char** argv) {
  if (argc != 4) {
    cerr << argv[0] << " <training.txt> <output.lm> <nsamples>\n\nEstimate a " << kORDER << "-gram HPYP LM and write it to a file\n100 is usually sufficient for <nsamples>\n";
    return 1;
  }
  MT19937 eng;
  string train_file = argv[1];
  string output_file = argv[2];
  {
    ifstream test(output_file);
    if (test.good()) {
      cerr << "File " << output_file << " appears to exist: please remove\n";
      return 1;
    }
  }
  int samples = atoi(argv[3]);
  assert(samples > 0);

  vector<vector<WordId> > corpus;
  set<WordId> vocabe, tv;
  const WordId kSOS = dict.Convert("<s>");
  const WordId kEOS = dict.Convert("</s>");
  cerr << "Reading corpus...\n";
  ReadFromFile(train_file, &dict, &corpus, &vocabe);
  cerr << "E-corpus size: " << corpus.size() << " sentences\t (" << vocabe.size() << " word types)\n";
  PYPLM<kORDER> lm(vocabe.size(), 1, 1, 1, 1);
  vector<WordId> ctx(kORDER - 1, kSOS);
  for (int sample=0; sample < samples; ++sample) {
    for (const auto& s : corpus) {
      ctx.resize(kORDER - 1);
      for (unsigned i = 0; i <= s.size(); ++i) {
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
  cerr << "Writing LM to " << output_file << " ...\n";
  ofstream ofile(output_file.c_str(), ios::out | ios::binary);
  if (!ofile.good()) {
    cerr << "Failed to open " << output_file << " for writing\n";
    return 1;
  }
  boost::archive::binary_oarchive oa(ofile);
  oa & dict;
  oa & lm;

  return 0;
}

