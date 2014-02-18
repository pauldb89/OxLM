// STL
#include <vector>
#include <iostream>
#include <fstream>

// Boost
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/archive/text_iarchive.hpp>

// Eigen
#include <Eigen/Core>

// Local
#include "cg/cnlm.h"
#include "corpus/corpus.h"
#include "cg/utils.h"

static const char *REVISION = "$Rev: 5 $";

// Namespaces
using namespace boost;
using namespace boost::program_options;
using namespace std;
using namespace oxlm;
using namespace Eigen;

int main(int argc, char **argv) {
  cerr << "Get vectors from model: Copyright 2013 Edward Grefenstette, "
       << REVISION << '\n' << endl;

  ///////////////////////////////////////////////////////////////////////////////////////
  // Command line processing
  variables_map vm;

  // Command line processing
  options_description cmdline_specific("Command line specific options");
  cmdline_specific.add_options()
    ("help,h", "print help message");
  options_description generic("Allowed options");
  generic.add_options()
    ("model-in,m", value<string>(),
        "model to extract thetas from")
  ("source-vectors,s", 
    "Get source vectors (S matrix).")
  ("context-vectors,q", 
    "Get context vectors (Q matrix).")
  ("output-vectors,r", 
    "Get output vectors (R matrix).")
    ;
  options_description config_options, cmdline_options;
  config_options.add(generic);
  cmdline_options.add(generic).add(cmdline_specific);

  store(parse_command_line(argc, argv, cmdline_options), vm);
  notify(vm);
  ///////////////////////////////////////////////////////////////////////////////////////

  if (vm.count("help") || !vm.count("model-in") || (!vm.count("source-vectors") && !vm.count("context-vectors") && !vm.count("output-vectors"))) {
    cerr << "Missing arguments."
         << endl
         << cmdline_options 
         << endl;
    return 1;
  }

  if ((vm.count("source-vectors") && vm.count("context-vectors"))
      || (vm.count("source-vectors") && vm.count("output-vectors"))
      || (vm.count("context-vectors") && vm.count("output-vectors"))) {
    cerr << "Choose at most one of source-vectors, context-vectors, or output-vectors."
         << endl
         << cmdline_options 
         << endl;
    return 1;
  }

  cerr << "Loading model." << endl;

  CNLMBase model;
  std::ifstream f(vm["model-in"].as<string>().c_str());
  boost::archive::text_iarchive ar(f);
  ar >> model;

  cerr << "Model loaded." << endl;

  Dict label_dict;
  vector<Word> vocabulary;
  MatrixReal M;

  if (vm.count("source-vectors")){ 
    label_dict = model.source_label_set();
    vocabulary = label_dict.getVocab();
    M = model.S;
  }
  else if (vm.count("context-vectors")){ 
    label_dict = model.label_set();
    vocabulary = label_dict.getVocab();
    M = model.Q;
  }
  else if (vm.count("output-vectors")){ 
    label_dict = model.label_set();
    vocabulary = label_dict.getVocab();
    M = model.R;
  }
  
  for (vector<Word>::const_iterator i=vocabulary.cbegin(); i < vocabulary.end(); i++) {
    if (*i == "<s>" || *i=="</s>") continue;
    WordId wid = label_dict.Convert(*i);
    cout << *i << " " << M.row(wid) << endl;
  }

  return 0;
}
