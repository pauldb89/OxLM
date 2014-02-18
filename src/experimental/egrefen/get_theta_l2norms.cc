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

static const char *REVISION = "$Rev: 3 $";

// Namespaces
using namespace boost;
using namespace boost::program_options;
using namespace std;
using namespace oxlm;
using namespace Eigen;

int main(int argc, char **argv) {
  cerr << "Get l2 norm of hidden variables: Copyright 2013 Edward Grefenstette, "
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

  cerr << "Loading model." << endl;

  CNLMBase model;
  std::ifstream f(vm["model-in"].as<string>().c_str());
  boost::archive::text_iarchive ar(f);
  ar >> model;

  cerr << "Model loaded." << endl;

  string token;
  ifstream source_in(vm["source"].as<string>().c_str());

  MatrixReal S = model.S;
  Real source_l2 = model.config.source_l2_parameter;

  while (source_in >> token) {
    WordId sid = model.source_label_set().Convert(token);

    VectorReal r = S.row(sid);
    Real logPtheta = (-0.5*source_l2*r.squaredNorm());

    cout << token << "\t" << logPtheta << endl;
  }

  source_in.close();

  return 0;
}
