// STL
#include <vector>
#include <iostream>
#include <fstream>
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
#include "utils/conditional_omp.h"
#include "cg/additive-cnlm.h"
#include "corpus/corpus.h"

static const char *REVISION = "$Rev: 1 $";

// Namespaces
using namespace boost;
using namespace boost::program_options;
using namespace std;
using namespace oxlm;
using namespace Eigen;


typedef vector<WordId> Context;
typedef std::vector<int> IntLabels;
typedef Sentence Label;

Real getSentenceProb(Sentence& s, Label& l, AdditiveCNLM& model, double penalty);

int main(int argc, char **argv) {

  cerr << "Find paraphrases with trained neural monolingual translation models: Copyright 2013 Karl Moritz Hermann and Ed Grefenstette, "
    << REVISION << '\n' << endl;

  ///////////////////////////////////////////////////////////////////////////////////////
  // Command line processing
  variables_map vm;

  // Command line processing
  options_description cmdline_specific("Command line specific options");
  cmdline_specific.add_options()
    ("help,h", "print help message")
    ;
  options_description generic("Allowed options");
  generic.add_options()
    ("labels,l", value<string>(),
     "list of candidate outputs, candidates per sentence in one line, referring to line in targets")
    ("symbols,s", value<string>(),
     "symbols to generate from, one per line")
    ("targets,t", value<string>(), "target sentences, one per line")
    ("model-in,m", value<string>(),
     "model to generate from")
    ("no-sentence-predictions", "do not print sentence predictions for individual sentences")
    ("raw-scores", "print only raw scores")
    ("word-insertion-penalty", value<double>()->default_value(0.0),
     "word insertion penalty")
    ;
  options_description config_options, cmdline_options;
  config_options.add(generic);
  cmdline_options.add(generic).add(cmdline_specific);

  store(parse_command_line(argc, argv, cmdline_options), vm);
  notify(vm);
  ///////////////////////////////////////////////////////////////////////////////////////

  if (vm.count("help") || !vm.count("labels") || !vm.count("model-in") || !vm.count("symbols")) {
    cerr << cmdline_options << "\n";
    return 1;
  }

  double penalty = vm["word-insertion-penalty"].as<double>();

  AdditiveCNLM model;
  std::ifstream f(vm["model-in"].as<string>().c_str());
  boost::archive::text_iarchive ar(f);
  ar >> model;

  WordId end_id = model.label_set().Lookup("</s>");

  string line, token;
  int label;

  // Read in candidate sentences
  ifstream targets_in(vm["targets"].as<string>().c_str());
  std::vector<Sentence> target_sentences;
  while(getline(targets_in, line)) {
    Sentence s;
    s.clear();
    stringstream sentence_stream(line);
    while (sentence_stream >> token)
      s.push_back(model.label_set().Convert(token, true));
    s.push_back(end_id);
    target_sentences.push_back(s);
  }
  targets_in.close();

  // Read in candidate sentence pointers
  ifstream labels_in(vm["labels"].as<string>().c_str());
  std::vector<IntLabels> candidates_per_symbol;

  while(getline(labels_in, line)) {
    stringstream label_stream(line);
    IntLabels candidate_labels;
    while (label_stream >> label)
      candidate_labels.push_back(label);
    candidates_per_symbol.push_back(candidate_labels);
  }
  labels_in.close();

  vector<int> predictedLabels;

  // read in symbols and calculate sentence probabilities p(E_i|\theta_g)
  ifstream sentences_in(vm["symbols"].as<string>().c_str());
  int counter = 0;
  while(getline(sentences_in, line)) {
    // get a symbol
    Label symbol;
    symbol.clear();
    stringstream sentence_stream(line);
    while (sentence_stream >> token)
      symbol.push_back(model.source_label_set().Convert(token, true));
    // if (model.config.source_eos) symbol.push_back(end_id);

    // calculate probability of sentences given that symbol.
    vector<Real> labelCondProbs;
    IntLabels& candidates = candidates_per_symbol[counter];
    for(size_t c_i = 0; c_i < candidates.size(); ++c_i) {

      Real condSentenceProb = getSentenceProb(target_sentences[candidates[c_i]], symbol, model, penalty);

      Real condLabelProb = condSentenceProb;
      // + labelLogProb; // We have no prior probability on a label (=sentence).

      labelCondProbs.push_back(condLabelProb);
    }

    // get index of max label and return max label
    Real maxVal = labelCondProbs.at(0);
    size_t maxIndex = 0;
    if(vm.count("raw-scores")) cout << "Next sentence" << endl;
    for(size_t c_i = 0; c_i < labelCondProbs.size(); ++c_i)
    {
      if(vm.count("raw-scores")) cout << "Comp: " << c_i << ": " << labelCondProbs.at(c_i) << endl;
      if (labelCondProbs.at(c_i) > maxVal) {
        maxIndex = c_i;
        maxVal = labelCondProbs.at(c_i);
      }
    }
    predictedLabels.push_back(maxIndex);

    if(!vm.count("no-sentence-predictions") && !vm.count("raw-scores")) cout << line << " ||| " << maxIndex << endl;
    if(vm.count("raw-scores")) cout << maxIndex << endl;
    ++counter;
  }
  sentences_in.close();

  // Get accuracy
  size_t correct = 0;
  size_t total = 0;

  for(size_t i=0; i<predictedLabels.size(); i++) {
    if (predictedLabels.at(i) == 0) correct++;
    total++;
  }
  Real accuracy = static_cast<Real>(correct)/static_cast<Real>(total);

  cout << "#######################" << endl << "# Accuracy = " << accuracy << endl << "#######################" << endl;

  return 0;

}

Real getSentenceProb(Sentence& s, Label& l, AdditiveCNLM& model, double penalty) {

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
      sentence_p += log_prob + log(penalty);
    }
    else {
      //        cerr << "Ignoring word for s_i=" << s_i << endl;
    }

  }

  return sentence_p;

}
