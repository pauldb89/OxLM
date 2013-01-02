#include <iostream>
#include <unordered_map>
#include <cstdlib>

#include "hpyplm.h"
#include "corpus/corpus.h"

#include "pyp/boost_serializers.h"
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>

#define kORDER 3

using namespace std;
using namespace pyp;

int main(int argc, char** argv) {
  if (argc != 3) {
    cerr << argv[0] << " <input.lm> <test.txt>\n\nCompute perplexity of a " << kORDER << "-gram HPYP LM\n";
    return 1;
  }
  MT19937 eng;
  string lm_file = argv[1];
  string test_file = argv[2];

  PYPLM<kORDER> lm;
  //vector<unsigned> ctx(kORDER - 1, kSOS);

  cerr << "Reading LM from " << lm_file << " ...\n";
  ifstream ifile(lm_file.c_str(), ios::in | ios::binary);
  if (!ifile.good()) {
    cerr << "Failed to open " << lm_file << " for reading\n";
    return 1;
  }
  boost::archive::binary_iarchive ia(ifile);
  Dict dict;
  ia & dict;
  ia & lm;
  const unsigned max_iv = dict.max();
  const unsigned kSOS = dict.Convert("<s>");
  const unsigned kEOS = dict.Convert("</s>");
  set<unsigned> tv;
  vector<vector<unsigned> > test;
  ReadFromFile(test_file, &dict, &test, &tv);
  double llh = 0;
  unsigned cnt = 0;
  unsigned oovs = 0;
  vector<unsigned> ctx(kORDER - 1, kSOS);
  for (auto& s : test) {
    ctx.resize(kORDER - 1);
    for (unsigned i = 0; i <= s.size(); ++i) {
      unsigned w = (i < s.size() ? s[i] : kEOS);
      double lp = log(lm.prob(w, ctx)) / log(2);
      if (w >= max_iv) {
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

