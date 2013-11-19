// STL
#include <vector>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <random>
#include <assert.h>

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
#include "cg/additive-cnlm.h"
#include "corpus/corpus.h"

static const char *REVISION = "$Rev: 1 $";

// Namespaces
using namespace boost;
using namespace boost::program_options;
using namespace std;
using namespace oxlm;
using namespace Eigen;

int main(int argc, char **argv) {

  cerr << "Do stuff: Copyright 2013 Karl Moritz Hermann, "
    << REVISION << '\n' << endl;

  //////////////////////////////////////////////////////////////////////////////
  // Command line processing
  variables_map vm;

  // Command line processing
  options_description cmdline_specific("Command line specific options");
  cmdline_specific.add_options()
    ("help,h", "print help message")
    ;
  options_description generic("Allowed options");
  generic.add_options()
    ("input,i", value<string>(),
     "input, topic label followed by sentence symbol (C s1311), one per line")
    ("output,o", value<string>(),
     "output file, topic label followed by vector (C 1:0.1 2:0.3 ... 256:0.01), one per line")
    ("model-in,m", value<string>(),
     "model to generate from")
    ;
  options_description config_options, cmdline_options;
  config_options.add(generic);
  cmdline_options.add(generic).add(cmdline_specific);

  store(parse_command_line(argc, argv, cmdline_options), vm);
  notify(vm);
  //////////////////////////////////////////////////////////////////////////////

  if (vm.count("help") || !vm.count("input") || !vm.count("output") || !vm.count("model-in")) {
    cerr << cmdline_options << "\n";
    return 1;
  }

  AdditiveCNLM model;
  std::ifstream f(vm["model-in"].as<string>().c_str());
  boost::archive::text_iarchive ar(f);
  ar >> model;

  WordId end_id = model.label_set().Lookup("</s>");

  // read in sentences and assign labels
  ifstream symbols_in(vm["input"].as<string>().c_str());
  ofstream outfile(vm["output"].as<string>().c_str());
  string topic, word, line;
  VectorReal result = VectorReal::Zero(model.config.word_representation_size);
  while(getline(symbols_in, line)) {
    stringstream sentence_stream(line);
    sentence_stream >> topic >> word;

    WordId id = model.source_label_set().Lookup(word);
    assert(id > -1);
    result = model.S.row(id);
    outfile << topic;
    for (int i = 0; i < model.config.word_representation_size; ++i) {
       outfile << " " << i << ":" << result[i];
    }
    outfile << "\n";
  }
  symbols_in.close();
  outfile.close();

  return 0;
}
