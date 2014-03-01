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
#include "cg/cnlm.h"
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
    cerr << "Score sentences with a trained neural translation models: Copyright 2013 Phil Blunsom, "
         << REVISION << '\n' << endl;

    ///////////////////////////////////////////////////////////////////////////////////////
    // Command line processing
    variables_map vm;

    // Command line processing
    options_description cmdline_specific("Command line specific options");
    cmdline_specific.add_options()
    ("help,h", "print help message")
    ("config,c", value<string>(),
     "config file specifying additional command line options")
    ;
    options_description generic("Allowed options");
    generic.add_options()
    ("source,s", value<string>(),
     "corpus of sentences, one per line")
    ("target,t", value<string>(),
     "reference translations of the source sentences, one per line")
    ("threads", value<int>()->default_value(1),
     "number of worker threads.")
    ("model-in,m", value<string>(),
     "model to generate from")
    ("no-source-eos", "do not add end of sentence tag to source \
                       representations.")
    ("print-sentence-llh", "print the LLH of each sentence")
    ("print-corpus-ppl", "print perplexity of the corpus")
    ;
    options_description config_options, cmdline_options;
    config_options.add(generic);
    cmdline_options.add(generic).add(cmdline_specific);

    store(parse_command_line(argc, argv, cmdline_options), vm);
    if (vm.count("config") > 0) {
        ifstream config(vm["config"].as<string>().c_str());
        store(parse_config_file(config, cmdline_options), vm);
    }
    notify(vm);
    ///////////////////////////////////////////////////////////////////////////////////////

    if (vm.count("help") || !vm.count("source") || !vm.count("model-in") || !vm.count("target")) {
        cerr << cmdline_options << "\n";
        return 1;
    }

    omp_set_num_threads(vm["threads"].as<int>());

    CNLMBase model;
    std::ifstream f(vm["model-in"].as<string>().c_str());
    boost::archive::text_iarchive ar(f);
    ar >> model;
//  cerr << model.length_ratio << endl;

    WordId end_id = model.label_set().Lookup("</s>");

    //////////////////////////////////////////////
    // process the input sentences
    string line, token;
    ifstream source_in(vm["source"].as<string>().c_str());
    ifstream target_in(vm["target"].as<string>().c_str());

    int source_counter=0;
    int context_width = model.config.ngram_order-1;
    int tokens=0;
    WordId start_id = model.label_set().Lookup("<s>");
    Real pp=0.0;
    std::vector<WordId> context(context_width);

    while (getline(source_in, line)) {
        // read the sentence
        stringstream line_stream(line);
        Sentence s;
        while (line_stream >> token)
            s.push_back(model.source_label_set().Convert(token));
        if (!vm.count("no-source-eos"))
            s.push_back(end_id);

        Sentence t;
        assert(getline(target_in, line));
        stringstream target_line_stream(line);
        while (target_line_stream >> token)
            t.push_back(model.label_set().Convert(token));
        t.push_back(end_id);

        Real sentence_p=0.0;
        for (size_t t_i=0; t_i < t.size(); ++t_i) {
            WordId w = t.at(t_i);
            if(model.label_set().valid(w)) {
                int context_start = t_i - context_width;
                bool sentence_start = (t_i==0);

                for (int i=context_width-1; i>=0; --i) {
                    int j=context_start+i;
                    sentence_start = (sentence_start || j<0);
                    int v_i = (sentence_start ? start_id : t.at(j));

                    context.at(i) = v_i;
                }
                Real log_prob = model.log_prob(w, context, s, false, t_i);
                sentence_p += log_prob;
            }

            tokens++;

        }
        pp += sentence_p;

        if (vm.count("print-sentence-llh"))
            cout << sentence_p << endl;

        source_counter++;
    }
    source_in.close();
    target_in.close();
    //////////////////////////////////////////////

    if (vm.count("print-corpus-ppl"))
        cout << "Corpus Perplexity = " << exp(-pp/tokens) << endl;

    return 0;
}
