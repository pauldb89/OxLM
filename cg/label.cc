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
#include "cg/cnlm.h"
#include "corpus/corpus.h"

static const char *REVISION = "$Rev: 1 $";

// Namespaces
using namespace boost;
using namespace boost::program_options;
using namespace std;
using namespace oxlm;
using namespace Eigen;


typedef vector<WordId> Context;
typedef std::pair<std::string, Real> LabelLogProb;
typedef std::vector<LabelLogProb> LabelLogProbs;
typedef Sentence Label;

Real getSentenceProb(Sentence& s, Label& l, ConditionalNLM& model);

int main(int argc, char **argv) {

    cerr << "Assign labels with trained neural monolingual translation models: Copyright 2013 Ed Grefenstette, "
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
      ("labels,l", value<string>(),
          "list of labels and their probability, one per line")
      ("sentences,s", value<string>(),
          "sentences to be labelled, one per line")
      ("threads", value<int>()->default_value(1),
          "number of worker threads.")
      ("model-in,m", value<string>(),
          "model to generate from")
      ("reference,r", value<string>(), "reference labels, one per line")
//      ("print-sentence-llh", "print the LLH of each sentence")
//      ("print-corpus-ppl", "print perplexity of the corpus")
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

    if (vm.count("help") || !vm.count("labels") || !vm.count("model-in") || !vm.count("sentences")) {
      cerr << cmdline_options << "\n";
      return 1;
    }

    ConditionalNLM model;
    std::ifstream f(vm["model-in"].as<string>().c_str());
    boost::archive::text_iarchive ar(f);
    ar >> model;

    WordId end_id = model.label_set().Lookup("</s>");


    // Read in label probabilities
    ifstream labels_in(vm["labels"].as<string>().c_str());
    LabelLogProbs labelLogProbs;

    string label, line, token;
    Real probability;

    while (labels_in >> label >> probability) {
        LabelLogProb labelInfo = make_pair(label, probability);
        labelLogProbs.push_back(labelInfo);
    }
    labels_in.close();


    vector<string> predictedLabels;

    // read in sentences and assign labels
    ifstream sentences_in(vm["sentences"].as<string>().c_str());
    while(getline(sentences_in, line)) {
        // get a sentence
        Sentence s;
        s.clear();
        stringstream sentence_stream(line);
        while (sentence_stream >> token)
            s.push_back(model.label_set().Convert(token, true));
        s.push_back(end_id);

        // calculate probability of labels given
        vector<Real> labelCondProbs;
        for(size_t l_i = 0; l_i < labelLogProbs.size(); ++l_i) {

            // unpack label and probability
            LabelLogProb llpInfo = labelLogProbs.at(l_i);
            string labelText = llpInfo.first;
            Real labelLogProb = llpInfo.second;

            // set up label for model query
            Label l;
            l.push_back(model.source_label_set().Convert(labelText));
            if (model.config.source_eos) l.push_back(end_id);

            Real condSentenceProb = getSentenceProb(s, l, model);
            
            Real condLabelProb = condSentenceProb + labelLogProb;

            labelCondProbs.push_back(condLabelProb);
        }

        // get index of max label and return max label
        Real maxVal = labelCondProbs.at(0);
        size_t maxIndex = 0;
        for(size_t l_i = 1; l_i < labelCondProbs.size(); ++l_i)
        {
            if (labelCondProbs.at(l_i) > maxVal) maxIndex = l_i;
        }
        string maxLabel = labelLogProbs.at(maxIndex).first;
        predictedLabels.push_back(maxLabel);

        cout << line << " ||| " << maxLabel << endl;
    }
    sentences_in.close();

    // Load reference labels, if provided, and get accuracy
    if (vm.count("reference")) {
        ifstream reflabels_in(vm["reference"].as<string>().c_str());

        vector<string> referenceLabels;
        string referenceLabel;
        while (reflabels_in >> referenceLabel) referenceLabels.push_back(referenceLabel);
        assert(referenceLabels.size()==predictedLabels.size() && "Label list size mismatch!");

        size_t correct;
        size_t total;

        for(size_t i=0; i<referenceLabels.size(); i++) {
            if (referenceLabels.at(i) == predictedLabels.at(i)) correct++;
            total++;
        }
        Real accuracy = static_cast<Real>(correct)/static_cast<Real>(total);

        cout << "#######################" << endl << "# Accuracy = " << accuracy << endl << "#######################" << endl;

        reflabels_in.close();
    }

    return 0;

}

Real getSentenceProb(Sentence& s, Label& l, ConditionalNLM& model) {

    WordId start_id = model.label_set().Lookup("<s>");
    int context_width = model.config.ngram_order-1;
    std::vector<WordId> context(context_width);

    Real sentence_p=0.0;
    for (size_t s_i=0; s_i < s.size(); ++s_i) {
      WordId w = s.at(s_i);
      if(model.label_set().valid(w)) {
        int context_start = s_i - context_width;
        bool sentence_start = (s_i==0);
  
        for (int i=context_width-1; i>=0; --i) {
          int j=context_start+i;
          sentence_start = (sentence_start || j<0);
          int v_i = (sentence_start ? start_id : s.at(j));
  
          context.at(i) = v_i;
        }
  
        Real log_prob = model.log_prob(w, context, l, false, s_i);
        sentence_p += log_prob;
      }
      else {
        cerr << "Ignoring word for s_i=" << s_i << endl;
      }

   }

   return sentence_p;

}
