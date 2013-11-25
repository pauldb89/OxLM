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

// Local
#include "utils/conditional_omp.h"
#include "cg/additive-cnlm.h"
#include "corpus/corpus.h"

using namespace boost;
using namespace boost::program_options;
using namespace std;
using namespace oxlm;

static const char *REVISION = "$Rev: 1 $";

int main(int argc, char **argv) {

    cerr << "Read parameters from monolingual translation models: Copyright 2013 Ed Grefenstette, "
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
      ("model,m", value<string>(),
          "model to generate from")
      ;
    options_description config_options, cmdline_options;
    config_options.add(generic);
    cmdline_options.add(generic).add(cmdline_specific);

    store(parse_command_line(argc, argv, cmdline_options), vm);
    notify(vm);

    if (vm.count("help") || !vm.count("model")) {
      cerr << cmdline_options << "\n";
      return 1;
    }

    AdditiveCNLM model;
    std::ifstream f(vm["model"].as<string>().c_str());
    boost::archive::text_iarchive ar(f);
    ar >> model;

    ModelData config = model.config;

    cerr << "################################" << endl;
    cerr << "# Config Summary:" << endl;
    cerr << "# order = " << config.ngram_order << endl;
    cerr << "# step-size = " << config.step_size << endl;
    cerr << "# lambda = " << config.l2_parameter << endl;
    cerr << "# source-lambda = " << config.source_l2_parameter << endl;
    cerr << "# iterations = " << config.iteration_size << endl;
    cerr << "# threads = " << config.threads << endl;
    cerr << "# classes = " << config.classes << endl;
    cerr << "# diagonal = " << config.diagonal << endl;
    cerr << "# non-linear = " << config.nonlinear << endl;
    cerr << "# width = " << config.source_window_width << endl;
    cout << "################################" << endl << endl;

    return 0;
}

