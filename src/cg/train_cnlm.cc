// STL
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <time.h>
#include <math.h>

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


void learn(const variables_map& vm, ModelData& config);
void cache_data(int start, int end, const vector<size_t>& indices, TrainingInstances &result);
Real perplexity(const AdditiveCNLM& model, const vector<Sentence>& test_source_corpus, const vector<Sentence>& test_target_corpus);
void freq_bin_type(const std::string &corpus, int num_classes, std::vector<int>& classes, Dict& dict, VectorReal& class_bias);
void classes_from_file(const std::string &class_file, vector<int>& classes, Dict& dict, VectorReal& class_bias);


int main(int argc, char **argv) {
  cout << "Online training for neural translation models: Copyright 2013 Phil Blunsom, "
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
        "corpus of sentences, one per line")
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
        "file containing word to class mappings in the format <class> <word> <frequence>.")
    ("window", value<int>()->default_value(-1),
        "Width of window of source words conditioned on.")
    ("no-source-eos", "do not add end of sentence tag to source representations.")
    ("replace-source-dict", "replace the source dictionary of a loaded model with a new one.")
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
  ///////////////////////////////////////////////////////////////////////////////////////

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

  learn(vm, config);

  return 0;
}


void learn(const variables_map& vm, ModelData& config) {
  vector<Sentence> target_corpus, source_corpus, test_target_corpus, test_source_corpus;
  Dict target_dict, source_dict;
  WordId end_id = target_dict.Convert("</s>");
  size_t num_training_instances=0, num_test_instances=0;

  //////////////////////////////////////////////
  // separate the word types into classes using
  // frequency binning
  vector<int> classes;
  VectorReal class_bias = VectorReal::Zero(config.classes);
  if (vm.count("class-file")) {
    cerr << "--class-file set, ignoring --classes." << endl;
    classes_from_file(vm["class-file"].as<string>(), classes, target_dict, class_bias);
    config.classes = classes.size()-1;
  }
  else
    freq_bin_type(vm["target"].as<string>(), config.classes, classes, target_dict, class_bias);
  //////////////////////////////////////////////

  //////////////////////////////////////////////
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
  //////////////////////////////////////////////

  //////////////////////////////////////////////
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
        assert(!"Word found in training target corpus, which wasn't encountered in originally trained and loaded model.");
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
      // Add words if the source dict is not frozen or if replace_source_dict is
      // set in which case the source_dict is new and hence unfrozen.
      WordId w = source_dict.Convert(token, (frozen_model && !replace_source_dict));
      if (w < 0) {
        cerr << token << " " << w << endl;
        assert(!"Word found in training source corpus, which wasn't encountered in originally trained and loaded model.");
      }
      s.push_back(w);
    }
    if (config.source_eos)
      s.push_back(end_id);
  }
  source_in.close();
  //////////////////////////////////////////////

  //////////////////////////////////////////////
  // read the test sentences
  bool have_test = vm.count("test-source");
  if (have_test) {
    ifstream test_source_in(vm["test-source"].as<string>().c_str());
    while (getline(test_source_in, line)) {
      stringstream line_stream(line);
      Sentence tokens;
      test_source_corpus.push_back(Sentence());
      Sentence& s = test_source_corpus.back();
      while (line_stream >> token) {
        WordId w = source_dict.Convert(token, true);
        if (w < 0) {
          cerr << token << " " << w << endl;
          assert(!"Unknown word found in test source corpus.");
        }
        s.push_back(w);
      }
      if (config.source_eos)
        s.push_back(end_id);
    }
    test_source_in.close();

    ifstream test_target_in(vm["test-target"].as<string>().c_str());
    while (getline(test_target_in, line)) {
      stringstream line_stream(line);
      Sentence tokens;
      test_target_corpus.push_back(Sentence());
      Sentence& s = test_target_corpus.back();
      while (line_stream >> token) {
        WordId w = target_dict.Convert(token, true);
        if (w < 0) {
          cerr << token << " " << w << endl;
          assert(!"Unknown word found in test target corpus.");
        }
        s.push_back(w);
      }
      s.push_back(end_id);
      num_test_instances += s.size();
    }
    test_target_in.close();
  }
  //////////////////////////////////////////////

  assert (source_corpus.size() == target_corpus.size());
  assert (test_source_corpus.size() == test_target_corpus.size());

  //////////////////////////////////////////////
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
    model.length_ratio += (Real(source_corpus.at(s).size()) / Real(target_corpus.at(s).size()));
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

  //////////////////////////////////////////////
  // Model training.

  VectorReal adaGrad = VectorReal::Zero(model.num_weights());
  VectorReal global_gradient(model.num_weights());
  Real av_f=0.0;
  Real pp=0;

  #pragma omp parallel shared(global_gradient, pp, av_f)
  {
    Real* gradient_data = new Real[model.num_weights()];
    AdditiveCNLM::WeightsType gradient(gradient_data, model.num_weights());

    size_t minibatch_size = vm["minibatch-size"].as<int>();

    #pragma omp master
    {
      cerr << endl << fixed << setprecision(2);
      cerr << " |" << setw(target_corpus.size()/(minibatch_size*100)+8) << " ITERATION";
      cerr << setw(11) << "TIME (s)" << setw(10) << "-LLH";
      if (vm.count("test-source"))
        cerr << setw(13) << "HELDOUT PPL";
      cerr << " |" << endl;
    }

    for (int iteration=0; iteration < vm["iterations"].as<int>(); ++iteration) {
      //clock_t iteration_start=clock();
      time_t iteration_start=time(0);
      #pragma omp master
      {
        av_f=0.0;
        pp=0.0;
//        cout << "Iteration " << iteration << ": "; cout.flush();

        if (vm.count("randomise"))
          std::random_shuffle(training_indices.begin(), training_indices.end());
      }

      TrainingInstances training_instances;
      Real step_size = vm["step-size"].as<float>();

      #pragma omp master
      cerr << " |" << setw(6) << iteration << " ";

      size_t minibatch_counter=0;
      for (size_t start=0; start < target_corpus.size() && (int)start < vm["instances"].as<int>(); ++minibatch_counter) {
        #pragma omp barrier

        size_t end = min(target_corpus.size(), start + minibatch_size);

        #pragma omp master
        global_gradient.setZero();

        gradient.setZero();
        Real l2 = config.l2_parameter*(end-start)/Real(target_corpus.size());
        Real l1 = config.l1_parameter*(end-start)/Real(target_corpus.size());
        Real l2_source = config.source_l2_parameter*(end-start)/Real(target_corpus.size());

        cache_data(start, end, training_indices, training_instances);
        Real f = model.gradient(source_corpus, target_corpus, training_instances, l2, l2_source, gradient);
        #pragma omp critical
        {
          global_gradient += gradient;
          av_f += f;
        }
        #pragma omp barrier

        #pragma omp master
        {
          // l2 regulariser contributions. Not very efficient
          //av_f += 0.5*l2*(model.C.squaredNorm() + model.R.squaredNorm() +
          //                    model.Q.squaredNorm() + model.B.squaredNorm() +
          //                    model.F.squaredNorm() + model.FB.squaredNorm());
          //av_f += 0.5*source_l2*(model.T.squaredNorm() + model.S.squaredNorm());
          //
          //av_f += (0.5*l2*model.W.squaredNorm());

          //global_gradient.array() += (l2 * model.W.array());

          adaGrad.array() += global_gradient.array().square();
          for (int w=0; w<model.num_weights(); ++w) {
            if (adaGrad(w)) {
              if (l1 > 0.0) {
                Real scale = step_size / sqrt(adaGrad(w));
                Real w1 = model.W(w) - scale*global_gradient(w);
                Real w2 = max(Real(0.0), abs(w1) - scale*l1);
                global_gradient(w) = w1 >= 0.0 ? w1 + w2 : w1 - w2;
              }
              else
                global_gradient(w) = (step_size*global_gradient(w)/ sqrt(adaGrad(w)));
            }
          }

          // Set unwanted weights to zero.
          // Map parameters using model, then set to zero.
          CNLMBase::WordVectorsType g_R(0,0,0), g_Q(0,0,0), g_F(0,0,0), g_S(0,0,0);
          CNLMBase::ContextTransformsType g_C, g_T;
          CNLMBase::WeightsType g_B(0,0), g_FB(0,0);
          Real* ptr = global_gradient.data();
          model.map_parameters(ptr, g_R, g_Q, g_F, g_C, g_B, g_FB, g_S, g_T);

          if (!config.updates.T) { for (auto g_t : g_T) g_t.setZero(); }
          if (!config.updates.C) { for (auto g_c : g_C) g_c.setZero(); }
          if (!config.updates.S) g_S.setZero();
          if (!config.updates.R) g_R.setZero();
          if (!config.updates.Q) g_Q.setZero();
          if (!config.updates.F) g_F.setZero();
          if (!config.updates.FB) g_FB.setZero();
          if (!config.updates.B) g_B.setZero();

          // Apply gradients to model.
          model.W -= global_gradient;

          if (minibatch_counter % 100 == 0) { cerr << "."; cout.flush(); }
        }
        if (l1 > 0.0) av_f += (l1 * model.W.lpNorm<1>());

        start += minibatch_size;
      }
     // #pragma omp master
//      cerr << endl;

      //Real iteration_time = (clock()-iteration_start) / (Real)CLOCKS_PER_SEC;
      int iteration_time = difftime(time(0),iteration_start);
      if (vm.count("test-source")) {
        Real local_pp = perplexity(model, test_source_corpus, test_target_corpus);

        #pragma omp critical
        { pp += local_pp; }
        #pragma omp barrier
      }

      #pragma omp master
      {
//          cerr << pp << " " << num_test_instances << " " << pp/num_test_instances;
        pp = exp(-pp/num_test_instances);
        cerr << setw(11) << iteration_time << setw(10) << av_f/num_training_instances;
        if (vm.count("test-source")) {
          cerr << setw(13) << pp;
        }
        cerr << " |" << endl;

        //cerr << " T norms:";
        //for (auto t : model.T)
        //  cerr << " " << t.norm();
        //cerr << endl;
      }
    }
  }

  if (vm.count("model-out")) {
    cout << "Writing trained model to " << vm["model-out"].as<string>() << endl;
    std::ofstream f(vm["model-out"].as<string>().c_str());
    boost::archive::text_oarchive ar(f);
    ar << model;
  }
}


void cache_data(int start, int end, const vector<size_t>& indices, TrainingInstances &result) {
  assert (start>=0 && start < end);

  size_t thread_num = omp_get_thread_num();
  size_t num_threads = omp_get_num_threads();

  result.clear();
  result.reserve((end-start)/num_threads);

  for (int s = start+thread_num; s < end; s += num_threads) {
    result.push_back(indices.at(s));
  }
}


Real perplexity(const AdditiveCNLM& model, const vector<Sentence>& test_source_corpus, const vector<Sentence>& test_target_corpus) {
  Real p=0.0;

  int context_width = model.config.ngram_order-1;
  int tokens=0;
  WordId start_id = model.label_set().Lookup("<s>");

//  #pragma omp master
//  cerr << "Calculating perplexity for " << test_target_corpus.size() << " sentences";

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
      Real log_prob = model.log_prob(w, context, test_source_corpus.at(s), false, t_i);
      p += log_prob;

//      #pragma omp master
//      if (tokens % 1000 == 0) { cerr << "."; cerr.flush(); }

      tokens++;
    }
  }
//  #pragma omp master
//  cerr << endl;

  return p;
}


void freq_bin_type(const std::string &corpus, int num_classes, vector<int>& classes, Dict& dict, VectorReal& class_bias) {
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
      int w_id = tmp_dict.insert(make_pair(token,tmp_dict.size())).first->second;
      assert (w_id <= int(counts.size()));
      if (w_id == int(counts.size())) counts.push_back( make_pair(token, 1) );
      else                            counts[w_id].second += 1;
      sum++;
    }
    eos_sum++;
  }

  sort(counts.begin(), counts.end(),
       [](const pair<string,int>& a, const pair<string,int>& b) -> bool { return a.second > b.second; });

  classes.clear();
  classes.push_back(0);

  classes.push_back(2);
  class_bias(0) = log(eos_sum);
  int bin_size = sum / (num_classes-1);

//  int bin_size = counts.size()/(num_classes);

  int mass=0;
  for (int i=0; i < int(counts.size()); ++i) {
    WordId id = dict.Convert(counts.at(i).first);

//    if ((mass += 1) > bin_size) {

    if ((mass += counts.at(i).second) > bin_size) {
      bin_size = (sum -= mass) / (num_classes - classes.size());
      class_bias(classes.size()-1) = log(mass);


//      class_bias(classes.size()-1) = 1;

      classes.push_back(id+1);

//      cerr << " " << classes.size() << ": " << classes.back() << " " << mass << endl;
      mass=0;
    }
  }
  if (classes.back() != int(dict.size()))
    classes.push_back(dict.size());

//  cerr << " " << classes.size() << ": " << classes.back() << " " << mass << endl;
  class_bias.array() -= log(eos_sum+sum);

  cerr << "Binned " << dict.size() << " types in " << classes.size()-1 << " classes with an average of "
       << float(dict.size()) / float(classes.size()-1) << " types per bin." << endl;
  in.close();
}


void classes_from_file(const std::string &class_file, vector<int>& classes, Dict& dict, VectorReal& class_bias) {
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
//      cerr << " " << classes.size() << ": " << classes.back() << " " << mass << endl;
      mass=0;
    }
    prev_class_str=class_str;
  }

  class_freqs.push_back(log(mass));
  classes.push_back(dict.size());
//  cerr << " " << classes.size() << ": " << classes.back() << " " << mass << endl;

  class_bias = VectorReal::Zero(class_freqs.size());
  for (size_t i=0; i<class_freqs.size(); ++i)
    class_bias(i) = class_freqs.at(i) - log(total_mass);

  cerr << "Read " << dict.size() << " types in " << classes.size()-1 << " classes with an average of "
       << float(dict.size()) / float(classes.size()-1) << " types per bin." << endl;

  in.close();
}
