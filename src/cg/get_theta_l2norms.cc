// STL
#include <vector>
#include <iostream>
#include <fstream>
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
#include "utils/conditional_omp.h"
#include "cg/additive-cnlm.h"
#include "corpus/corpus.h"

static const char *REVISION = "$Rev: 248 $";

// Namespaces
using namespace boost;
using namespace boost::program_options;
using namespace std;
using namespace oxlm;
using namespace Eigen;


typedef vector<WordId> Context;


int main(int argc, char **argv) {
  cerr << "Get l2 norm of hidden variables: Copyright 2013 Edward Grefenstette, "
       << REVISION << '\n' << endl;

  ///////////////////////////////////////////////////////////////////////////////////////
  // Command line processing
  variables_map vm;

  // Command line processing
  options_description cmdline_specific("Command line specific options");
  cmdline_specific.add_options()
    ("help,h", "print help message")
  options_description generic("Allowed options");
  generic.add_options()
    ("source,s", value<string>(),
        "corpus of theta-values, one per line")
    ("model-in,m", value<string>(),
        "model to generate from")
    ;
  options_description config_options, cmdline_options;
  config_options.add(generic);
  cmdline_options.add(generic).add(cmdline_specific);

  store(parse_command_line(argc, argv, cmdline_options), vm);
  notify(vm);
  ///////////////////////////////////////////////////////////////////////////////////////

  if (vm.count("help") || !vm.count("source") || !vm.count("model-in")) {
    cerr << cmdline_options << "\n";
    return 1;
  }

  AdditiveCNLM model;
  std::ifstream f(vm["model-in"].as<string>().c_str());
  boost::archive::text_iarchive ar(f);
  ar >> model;

  //////////////////////////////////////////////
  // process the input sentences
  string line, token;
  ifstream source_in(vm["source"].as<string>().c_str());

  int source_counter=0;
  int context_width = model.config.ngram_order-1;
  int tokens=0;

  WordId start_id = model.label_set().Lookup("<s>");
  Real pp=0.0;
  std::vector<WordId> context(context_width);

  cout << model.S;

  // while (cin >> token) {
  //   // read the sentence
  //   WordId label = model.source_label_set().Convert(token);

  //   cout << token << "\t" << l2norm << endl;
  // }
  source_in.close();
  target_in.close();
  //////////////////////////////////////////////

  if (vm.count("print-corpus-ppl"))
    cout << "Corpus Perplexity = " << exp(-pp/tokens) << endl;

  return 0;
}
