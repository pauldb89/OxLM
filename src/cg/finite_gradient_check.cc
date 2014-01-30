// STL
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <time.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cmath>

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
#include "corpus/alignment.h"

static const char *REVISION = "$Rev: 248 $";

// Namespaces
using namespace boost;
using namespace boost::program_options;
using namespace std;
using namespace oxlm;
using namespace Eigen;

void gradient_check(const variables_map& vm, ModelData& config, const Real epsilon);
void cache_data(int start, int end, const vector<size_t>& indices,
                TrainingInstances &result);
Real log_likelihood(const AdditiveCNLM& model,
                    const vector<Sentence>& test_source_corpus,
                    const vector<Sentence>& test_target_corpus);
void freq_bin_type(const std::string &corpus, int num_classes,
                   std::vector<int>& classes,
                   Dict& dict, VectorReal& class_bias);
void classes_from_file(const std::string &class_file, vector<int>& classes,
                       Dict& dict, VectorReal& class_bias);


int main(int argc, char **argv) {
  cout << "Online training for neural translation models: \
           Copyright 2013 Phil Blunsom, "
       << REVISION << '\n' << endl;

  //////////////////////////////////////////////////////////////////////////////
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
        "corpus of sentences, one per line")
    ("alignment,a", value<string>(),
        "Moses style alignment of source and target")
    ("test-source", value<string>(),
        "corpus of test sentences to be evaluated at each iteration")
    ("test-target", value<string>(),
        "corpus of test sentences to be evaluated at each iteration")
    ("iterations", value<int>()->default_value(10),
        "number of passes through the data")
    ("minibatch-size", value<int>()->default_value(100),
        "number of sentences per minibatch")
    ("instances", value<int>()->default_value(std::numeric_limits<int>::max()),
        "training instances per iteration")
    ("order,n", value<int>()->default_value(3),
        "ngram order")
    ("model-in,m", value<string>(),
        "initial model")
    ("model-out,o", value<string>()->default_value("model"),
        "base filename of model output files")
    ("l2,r", value<float>()->default_value(0.0),
        "l2 regularisation strength parameter")
    ("l1", value<float>()->default_value(0.0),
        "l1 regularisation strength parameter")
    ("source-l2", value<float>(),
        "source regularisation strength parameter")
    ("dump-frequency", value<int>()->default_value(0),
        "dump model every n minibatches.")
    ("word-width", value<int>()->default_value(100),
        "Width of word representation vectors.")
    ("threads", value<int>()->default_value(1),
        "number of worker threads.")
    ("step-size", value<float>()->default_value(1.0),
        "SGD batch stepsize, it is normalised by the number of minibatches.")
    ("classes", value<int>()->default_value(100),
        "number of classes for factored output.")
    ("class-file", value<string>(),
        "file containing word to class mappings in the format: \
         <class> <word> <frequence>.")
    ("window", value<int>()->default_value(-1),
        "Width of window of source words conditioned on.")
    ("no-source-eos", "do not add end of sentence tag to source \
                       representations.")
    ("replace-source-dict", "replace the source dictionary of a loaded model \
                             with a new one.")
    ("verbose,v", "print perplexity for each sentence (1) or input token (2) ")
    ("randomise", "visit the training tokens in random order.")
    ("diagonal-contexts", "Use diagonal context matrices (usually faster).")
    ("non-linear", "use a non-linear hidden layer.")
    ("updateT", value<bool>()->default_value(true), "update T weights?")
    ("updateS", value<bool>()->default_value(true), "update S weights?")
    ("updateC", value<bool>()->default_value(true), "update C weights?")
    ("updateR", value<bool>()->default_value(true), "update R weights?")
    ("updateQ", value<bool>()->default_value(true), "update Q weights?")
    ("updateF", value<bool>()->default_value(true), "update F weights?")
    ("updateFB", value<bool>()->default_value(true), "update FB weights?")
    ("updateB", value<bool>()->default_value(true), "update B weights?")
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
  //////////////////////////////////////////////////////////////////////////////

  if (vm.count("help") || !vm.count("source") || !vm.count("target")) {
    cout << cmdline_options << "\n";
    return 1;
  }

  ModelData config;
  config.l1_parameter = vm["l1"].as<float>();
  config.l2_parameter = vm["l2"].as<float>();
  if (vm.count("source-l2"))
    config.source_l2_parameter = vm["source-l2"].as<float>();
  else
    config.source_l2_parameter = config.l2_parameter;
  config.word_representation_size = vm["word-width"].as<int>();
  config.threads = vm["threads"].as<int>();
  config.ngram_order = vm["order"].as<int>();
  config.verbose = vm.count("verbose");
  config.classes = vm["classes"].as<int>();
  config.diagonal = vm.count("diagonal-contexts");
  config.nonlinear= vm.count("non-linear");
  config.source_window_width = vm["window"].as<int>();
  config.source_eos = !vm.count("no-source-eos");

  Bools updates;
  updates.T = vm["updateT"].as<bool>();
  updates.S = vm["updateS"].as<bool>();
  updates.C = vm["updateC"].as<bool>();
  updates.R = vm["updateR"].as<bool>();
  updates.Q = vm["updateQ"].as<bool>();
  updates.F = vm["updateF"].as<bool>();
  updates.FB = vm["updateFB"].as<bool>();
  updates.B = vm["updateB"].as<bool>();
  config.updates = updates;

  cerr << "################################" << endl;
  cerr << "# Config Summary" << endl;
  for (variables_map::iterator iter = vm.begin(); iter != vm.end(); ++iter)
  {
    cerr << "# " << iter->first << " = ";
    const ::std::type_info& type = iter->second.value().type() ;
    if ( type == typeid( ::std::string ) )
      cerr << iter->second.as<string>() << endl;
    if ( type == typeid( int ) )
      cerr << iter->second.as<int>() << endl;
    if ( type == typeid( double ) )
      cerr << iter->second.as<double>() << endl;
    if ( type == typeid( float ) )
      cerr << iter->second.as<float>() << endl;
    if ( type == typeid( bool ) )
      cerr << (iter->second.as<bool>() ? "true" : "false") << endl;
  }
  cerr << "################################" << endl;

  omp_set_num_threads(config.threads);

  // learn(vm, config);
  gradient_check(vm, config, 0.0001);

  return 0;
}


void gradient_check(const variables_map& vm, ModelData& config, const Real epsilon) {
  vector<Sentence> target_corpus, source_corpus;
  vector<Sentence> test_target_corpus, test_source_corpus;
  Dict target_dict, source_dict;
  WordId end_id = target_dict.Convert("</s>");
  size_t num_training_instances=0, num_test_instances=0;


  //////////////////////////////////////////////////////////////////////////////
  // separate the word types into classes using
  // frequency binning
  vector<int> classes;
  VectorReal class_bias = VectorReal::Zero(config.classes);
  if (vm.count("class-file")) {
    cerr << "--class-file set, ignoring --classes." << endl;
    classes_from_file(vm["class-file"].as<string>(), classes, target_dict,
                      class_bias);
    config.classes = classes.size()-1;
  }
  else
    freq_bin_type(vm["target"].as<string>(), config.classes, classes,
                  target_dict, class_bias);
  //////////////////////////////////////////////////////////////////////////////


  //////////////////////////////////////////////////////////////////////////////
  // create and or load the model.
  // If we do not load, we need to update some aspects of the model later, for
  // instance push the updated dictionaries. If we do load, we need up modify
  // aspects of the configuration (esp. weight update settings).
  AdditiveCNLM model(config, source_dict, target_dict, classes);
  bool frozen_model = false;
  bool replace_source_dict = false;
  if (vm.count("replace-source-dict")) {
    assert(vm.count("model-in"));
    replace_source_dict = true;
  }

  if (vm.count("model-in")) {
    std::ifstream f(vm["model-in"].as<string>().c_str());
    boost::archive::text_iarchive ar(f);
    ar >> model;
    target_dict = model.label_set();
    if(!replace_source_dict)
      source_dict = model.source_label_set();
    // Set dictionary update to false and freeze model parameters in general.
    frozen_model = true;
    // Adjust config.update parameter, as this is dependent on the specific
    // training run and not on the model per se.
    model.config.updates = config.updates;
  }
  //////////////////////////////////////////////////////////////////////////////


  //////////////////////////////////////////////////////////////////////////////
  // read the training sentences
  ifstream target_in(vm["target"].as<string>().c_str());
  string line, token;
  // read the target
  while (getline(target_in, line)) {
    stringstream line_stream(line);
    target_corpus.push_back(Sentence());
    Sentence& s = target_corpus.back();
    while (line_stream >> token) {
      WordId w = target_dict.Convert(token, frozen_model);
      if (w < 0) {
        cerr << token << " " << w << endl;
        assert(!"Word found in training target corpus, which wasn't \
                 encountered in originally trained and loaded model.");
      }
      s.push_back(w);
    }
    s.push_back(end_id);
    num_training_instances += s.size();
  }
  target_in.close();

  // read the source
  ifstream source_in(vm["source"].as<string>().c_str());
  while (getline(source_in, line)) {
    stringstream line_stream(line);
    source_corpus.push_back(Sentence());
    Sentence& s = source_corpus.back();
    while (line_stream >> token) {
      // Add words if the source dict is not frozen or if replace_source_dict
      // is set in which case the source_dict is new and hence unfrozen.
      WordId w = source_dict.Convert(token,
                                     (frozen_model && !replace_source_dict));
      if (w < 0) {
        cerr << token << " " << w << endl;
        assert(!"Word found in training source corpus, which wasn't \
                 encountered in originally trained and loaded model.");
      }
      s.push_back(w);
    }
    if (config.source_eos)
      s.push_back(end_id);
  }
  source_in.close();
  //////////////////////////////////////////////////////////////////////////////


  assert (source_corpus.size() == target_corpus.size());

  //////////////////////////////////////////////////////////////////////////////
  // read the alignment
  AlignmentPtr alignment;
  if (vm.count("alignment")) {
    ifstream alignment_in(vm["alignment"].as<string>().c_str());
    read_alignment(alignment_in, alignment);
  }


  //////////////////////////////////////////////////////////////////////////////
  // Non-frozen model means we just learned a (new) dictionary. This requires
  // re-initializing the model using those dictionaries.
  if(!frozen_model) {
    model.reinitialize(config, source_dict, target_dict, classes);
    cerr << "(Re)initializing model based on training data." << endl;
  }
  else if(replace_source_dict) {
    model.expandSource(source_dict);
    cerr << "Replacing source dictionary based on training data." << endl;
  }

  if(!frozen_model)
    model.FB = class_bias;

  for (size_t s = 0; s < source_corpus.size(); ++s)
    model.length_ratio += (Real(source_corpus.at(s).size())
                           / Real(target_corpus.at(s).size()));
  model.length_ratio /= Real(source_corpus.size());

  vector<size_t> training_indices(target_corpus.size());
  VectorReal unigram = VectorReal::Zero(model.labels());
  for (size_t i = 0; i < training_indices.size(); i++) {
    for (size_t j = 0; j < target_corpus.at(i).size(); j++)
      unigram(target_corpus.at(i).at(j)) += 1;
    training_indices[i] = i;
  }
  if(!frozen_model)
    model.B = ((unigram.array()+1.0)/(unigram.sum()+unigram.size())).log();


  //////////////////////////////////////////////////////////////////////////////
  // Model training.
  VectorReal adaGrad = VectorReal::Zero(model.num_weights());
  VectorReal global_gradient(model.num_weights());
  Real av_f=0.0;
  Real pp=0;

  Real delta=0.0;

  #pragma omp parallel shared(global_gradient, pp, av_f)
  {
    Real* gradient_data = new Real[model.num_weights()];
    AdditiveCNLM::WeightsType gradient(gradient_data, model.num_weights());

    #pragma omp master
    {
      av_f=0.0;
      pp=0.0;
      if (vm.count("randomise"))
        std::random_shuffle(training_indices.begin(), training_indices.end());
    }
    TrainingInstances training_instances;
    Real step_size = vm["step-size"].as<float>();
    size_t start = 0;
    size_t end = target_corpus.size();
    #pragma omp master
      global_gradient.setZero();
    gradient.setZero();
    Real l2 = config.l2_parameter*(end-start)/Real(target_corpus.size());
    Real l1 = config.l1_parameter*(end-start)/Real(target_corpus.size());
    Real l2_source = config.source_l2_parameter
      * (end-start)/Real(target_corpus.size());

    cache_data(start, end, training_indices, training_instances);
    Real f = model.gradient(source_corpus, target_corpus,
                            training_instances, l2, l2_source, gradient);
    // This is the gradient and error per thread.
#pragma omp critical
    {
      global_gradient += gradient;
      av_f += f;
    }
    if (l1 > 0.0) av_f += (l1 * model.W.lpNorm<1>());

#pragma omp barrier

    // We have the gradients (global_gradient) and error (av_f) now for the
    // standard case. Now, iterate over all variables and compare gradient with
    // change in error.
    // WTF, Ed: We're multithreading here!


    for (size_t param_index = 0; param_index < model.m_data_size; ++param_index) {
      #pragma omp single
        delta = 0.0;
      model.m_data[param_index] += epsilon; // Parameter + epsilon
      Real f_plus_epsilon = model.gradient(source_corpus, target_corpus,
                             training_instances, l2, l2_source, gradient);
      model.m_data[param_index] -= 2*epsilon; // Parameter - epsilon
      Real f_minus_epsilon = model.gradient(source_corpus, target_corpus,
                                            training_instances, l2, l2_source, gradient);
      model.m_data[param_index] += epsilon; // Put paramater back to normal
      // Real delta = abs(f_plus_epsilon - f_minus_epsilon);
      #pragma omp critical
      {
        delta += f_plus_epsilon - f_minus_epsilon;
        if (l1 > 0.0) av_f += (l1 * model.W.lpNorm<1>());
      }
      #pragma omp single
      {
        delta = delta / (2.0 * epsilon);
        cout << "Delta at " << param_index << ": " << delta << " vs. grad " <<
          global_gradient[param_index] << endl;
      }
    }
  }
}


void cache_data(int start, int end, const vector<size_t>& indices,
                TrainingInstances &result) {
  assert (start>=0 && start < end);

  size_t thread_num = omp_get_thread_num();
  size_t num_threads = omp_get_num_threads();

  result.clear();
  result.reserve((end-start)/num_threads);

  for (int s = start+thread_num; s < end; s += num_threads) {
    result.push_back(indices.at(s));
  }
}


Real log_likelihood(const AdditiveCNLM& model,
                    const vector<Sentence>& test_source_corpus,
                    const vector<Sentence>& test_target_corpus) {
  Real p=0.0;

  int context_width = model.config.ngram_order-1;
  int tokens=0;
  WordId start_id = model.label_set().Lookup("<s>");

  std::vector<WordId> context(context_width);

  size_t thread_num = omp_get_thread_num();
  size_t num_threads = omp_get_num_threads();
  for (size_t s = thread_num; s < test_target_corpus.size(); s += num_threads) {
    const Sentence& sent = test_target_corpus.at(s);
    VectorReal s_rep = VectorReal::Zero(model.config.word_representation_size);
//    model.source_representation(test_source_corpus.at(s), s_rep);
    for (size_t t_i=0; t_i < sent.size(); ++t_i) {
      WordId w = sent.at(t_i);
      int context_start = t_i - context_width;
      bool sentence_start = (t_i==0);

      for (int i=context_width-1; i>=0; --i) {
        int j=context_start+i;
        sentence_start = (sentence_start || j<0);
        int v_i = (sentence_start ? start_id : sent.at(j));

        context.at(i) = v_i;
      }
//      Real log_prob = model.log_prob(w, context, s_rep);
      Real log_prob = model.log_prob(w, context, test_source_corpus.at(s),
                                     false, t_i);
      p += log_prob;

      tokens++;
    }
  }

  return p;
}


void freq_bin_type(const std::string &corpus, int num_classes,
                   vector<int>& classes, Dict& dict, VectorReal& class_bias) {
  ifstream in(corpus.c_str());
  string line, token;

  map<string,int> tmp_dict;
  vector< pair<string,int> > counts;
  int sum=0, eos_sum=0;
  string eos = "</s>";

  while (getline(in, line)) {
    stringstream line_stream(line);
    while (line_stream >> token) {
      if (token == eos) continue;
      int w_id = tmp_dict.insert(
          make_pair(token,tmp_dict.size())).first->second;
      assert (w_id <= int(counts.size()));
      if (w_id == int(counts.size())) counts.push_back( make_pair(token, 1) );
      else                            counts[w_id].second += 1;
      sum++;
    }
    eos_sum++;
  }

  sort(counts.begin(), counts.end(),
       [](const pair<string,int>& a, const pair<string,int>& b) ->
       bool { return a.second > b.second; });

  classes.clear();
  classes.push_back(0);

  classes.push_back(2);
  class_bias(0) = log(eos_sum);
  int bin_size = sum / (num_classes-1);

  int mass=0;
  for (int i=0; i < int(counts.size()); ++i) {
    WordId id = dict.Convert(counts.at(i).first);

    if ((mass += counts.at(i).second) > bin_size) {
      bin_size = (sum -= mass) / (num_classes - classes.size());
      class_bias(classes.size()-1) = log(mass);

      classes.push_back(id+1);

      mass=0;
    }
  }
  if (classes.back() != int(dict.size()))
    classes.push_back(dict.size());

  class_bias.array() -= log(eos_sum+sum);

  cerr << "Binned " << dict.size() << " types in " << classes.size()-1
       << " classes with an average of "
       << float(dict.size()) / float(classes.size()-1)
       << " types per bin." << endl;
  in.close();
}


void classes_from_file(const std::string &class_file, vector<int>& classes,
                       Dict& dict, VectorReal& class_bias) {
  ifstream in(class_file.c_str());

  vector<int> class_freqs(1,0);
  classes.clear();
  classes.push_back(0);
  classes.push_back(2);

  int mass=0, total_mass=0;
  string prev_class_str="", class_str="", token_str="", freq_str="";
  while (in >> class_str >> token_str >> freq_str) {
    int w_id = dict.Convert(token_str);

    int freq = lexical_cast<int>(freq_str);
    mass += freq;
    total_mass += freq;

    if (!prev_class_str.empty() && class_str != prev_class_str) {
      class_freqs.push_back(log(mass));
      classes.push_back(w_id+1);
      mass=0;
    }
    prev_class_str=class_str;
  }

  class_freqs.push_back(log(mass));
  classes.push_back(dict.size());

  class_bias = VectorReal::Zero(class_freqs.size());
  for (size_t i=0; i<class_freqs.size(); ++i)
    class_bias(i) = class_freqs.at(i) - log(total_mass);

  cerr << "Read " << dict.size() << " types in " << classes.size()-1
       << " classes with an average of "
       << float(dict.size()) / float(classes.size()-1)
       << " types per bin." << endl;

  in.close();
}
