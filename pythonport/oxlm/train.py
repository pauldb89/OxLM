import sys
import StringIO
import os
import argparse
import ConfigParser
import math
import operator
import datetime

import numpy as np
# import theano
# from theano import tensor as T

import config
import corpus
import cnlm


def main():
    args = __get_args()
    configuration = __get_configuration(args)
    learn(args, configuration)

def learn(args, configuration):
    if type(args) is not argparse.Namespace:
        raise TypeError("Expecting args to be of type Namespace. Type was: %s." % type(args).__name__)
    if type(configuration) is not config.ModelData:
        raise TypeError("Expecting configuration to be of type ModelData. Type was: %s." % type(configuration).__name__)

    target_corpus = []
    source_corpus = []
    test_target_corpus = []
    test_source_corpus = []

    target_dict = corpus.Dict()
    source_dict = corpus.Dict()
    end_id = target_dict.Convert("</s>")

    # ============================================================
    # separate the word types into classes using frequency binning
    # ============================================================

    classes, class_bias = load_classes(args, target_dict, configuration)

    # =========================================================================
    # create and or load the model.
    # If we do not load, we need to update some aspects of the model later, for
    # instance push the updated dictionaries. If we do load, we need up modify
    #  aspects of the configuration (esp. weight update settings).
    # =========================================================================

    model, frozen_model, replace_source_dict = load_model()

    # ===========================
    # read the training sentences
    # ===========================

    num_training_instances = read_training_sentences(args, target_dict, frozen_model, end_id, target_corpus)

    # ===============
    # read the source
    # ===============

    read_source_sentences(args, source_dict, frozen_model, replace_source_dict, configuration, end_id, source_corpus)

    # =======================
    # read the test sentences
    # =======================

     # bool have_test = vm.count("test-source");
    if args.test_source: # if (have_test) {
        num_test_instances = read_test_sentences(args, source_dict, target_dict, configuration, end_id, test_source_corpus, test_target_corpus)

    ##################### Sanity Check ######################
    assert len(source_corpus) == len(target_corpus)
    assert len(test_source_corpus) == len(test_target_corpus)
    #########################################################

    # ========================================================================
    # Non-frozen model means we just learned a (new) dictionary. This requires
    # re-initializing the model using those dictionaries.
    # ========================================================================

    reinitialise_model()

    # ===============
    # Model training.
    # ===============

    adaGrad = np.zeros((model.num_weights(), 1)) # VectorReal adaGrad = VectorReal::Zero(model.num_weights());
    global_gradient = [0.0]*model.num_weights() # VectorReal global_gradient(model.num_weights());
    av_f = 0.0 # Real av_f=0.0;
    pp = 0 # Real pp=0;

    dump_freq = int(args.dump_freq) # const int dump_freq = vm["dump-frequency"].as<int>();

    if dump_freq > 0:
        raise NotImplementedError #   const string partialdir = vm["model-out"].as<string>() + ".partial/";
        raise NotImplementedError #   mkdir(partialdir.c_str(), 0777); // notice that 777 is different than 0777

    gradient_data = [0.0] * model.num_weights() # Real* gradient_data = new Real[model.num_weights()];
    raise NotImplementedError # AdditiveCNLM::WeightsType gradient(gradient_data, model.num_weights());

    minibatch_size = int(args.minibatch_size) # size_t minibatch_size = vm["minibatch-size"].as<int>();

     # cerr << endl << fixed << setprecision(2);
     # cerr << " |" << setw(target_corpus.size()/(minibatch_size*100)+8)
     #      << " ITERATION";
     # cerr << setw(11) << "TIME (s)" << setw(10) << "-LLH";
     # if (vm.count("test-source"))
     #   cerr << setw(13) << "HELDOUT PPL";
     # cerr << " |" << endl;

    for iteration in range(int(args.iterations)): # for (int iteration=0; iteration < vm["iterations"].as<int>(); ++iteration) {
        iteration_start = datetime.datetime.now() #   time_t iteration_start=time(0);
        av_f = 0.0 #     av_f=0.0;
        pp = 0.0 #     pp=0.0;

        if args.randomise is not None: #     if (vm.count("randomise"))
            np.random.shuffle(training_indices) #       std::random_shuffle(training_indices.begin(), training_indices.end());

        training_instances = [] #   TrainingInstances training_instances;
        step_size = float(args.step_size) #   Real step_size = vm["step-size"].as<float>();

        sys.stderr.write(" |"+" "*6+str(iteration)+" ") #   cerr << " |" << setw(6) << iteration << " ";

        minibatch_counter = 0 #   size_t minibatch_counter=0;
        start = 0 #   for (size_t start=0; start < target_corpus.size()
        while(start < len(target_corpus) and int(start) < int(args.instances)): #        && (int)start < vm["instances"].as<int>(); ++minibatch_counter) {

            end = min(len(target_corpus), start + minibatch_size) #     size_t end = min(target_corpus.size(), start + minibatch_size);

            global_gradient *= 0 # global_gradient.setZero();

            gradient *= 0 # gradient.setZero();
            l2 = configuration.l2_parameter * (end-start) / float(len(target_corpus)) #  Real l2 = config.l2_parameter*(end-start)/Real(target_corpus.size());
            l1 = configuration.l2_parameter * (end-start) / float(len(target_corpus)) #  Real l1 = config.l1_parameter*(end-start)/Real(target_corpus.size());
            l2_source = configuration.source_l2_parameter * (end-start)/float(len(target_corpus)) # Real l2_source = config.source_l2_parameter * (end-start)/Real(target_corpus.size());

            raise NotImplementedError #     cache_data(start, end, training_indices, training_instances);
            f = model.gradient(source_corpus, target_corpus, training_instances, l2, l2_source, gradient) #     Real f = model.gradient(source_corpus, target_corpus, training_instances, l2, l2_source, gradient);

            global_gradient += gradient #     global_gradient += gradient;
            av_f += f #     av_f += f;
             #
            if l1 > 0.0:
                raise NotImplementedError #     if (l1 > 0.0) av_f += (l1 * model.W.lpNorm<1>());

            adaGrad += np.square(global_gradient) #      adaGrad.array() += global_gradient.array().square();


            for w in range(model.num_weights): #       for (int w=0; w<model.num_weights(); ++w) {
                if adaGrad[w]: #         if (adaGrad(w)) {
                    scale = step_size / math.sqrt(adaGrad[w]) #           Real scale = step_size / sqrt(adaGrad(w));
                    global_gradient[w] = scale * global_gradient[w] #           global_gradient(w) = scale * global_gradient(w);
                    if l1 > 0.0: #           if (l1 > 0.0) {
                        w1 = model.W[w] - global_gradient[w] #             Real w1 = model.W(w) - global_gradient(w);
                        w2 = max(0.0, abs(w1) - scale*l1) #             Real w2 = max(Real(0.0), abs(w1) - scale*l1);
                        if w1 >= 0.0:
                            global_gradient[w] =  model.W(w) - w2
                        else:
                            global_gradient[w] = model.W(w) + w2  # global_gradient(w) = w1 >= 0.0 ? model.W(w) - w2 : model.W(w) + w2;

             #       // Set unwanted weights to zero.
             #       // Map parameters using model, then set to zero.
             #       CNLMBase::WordVectorsType g_R(0,0,0), g_Q(0,0,0),
             #                                 g_F(0,0,0), g_S(0,0,0);
             #       CNLMBase::ContextTransformsType g_C, g_T;
             #       CNLMBase::WeightsType g_B(0,0), g_FB(0,0);
             #       Real* ptr = global_gradient.data();
             #       model.map_parameters(ptr, g_R, g_Q, g_F, g_C, g_B, g_FB, g_S, g_T);
             #
             #       // Apply gradients to model.
             #       model.W -= global_gradient;
             #
             #       if (minibatch_counter % 100 == 0) { cerr << "."; cout.flush(); }
             #
             #       if ((dump_freq > 0) && (minibatch_counter % dump_freq) == 0 ) {
             #         string partial_model_path = vm["model-out"].as<string>() + ".partial/"
             #                                                                  + "it" + std::to_string(iteration)
             #                                                                  + ".mb" + std::to_string(minibatch_counter)
             #                                                                  + ".model";
             #         cout << "Saving trained model from iteration " << iteration
             #                                                         << ", minibatch " << minibatch_counter
             #                                                         << " to " << partial_model_path << endl;
             #         cout.flush();
             #
             #         std::ofstream f(partial_model_path.c_str());
             #         boost::archive::text_oarchive ar(f);
             #         ar << model;
             #       }
             #
            start += minibatch_size #     start += minibatch_size;
            minibatch_counter += 1
         #   }

        iteration_time = (datetime.datetime.now() - iteration_start).c.total_seconds() #   int iteration_time = difftime(time(0),iteration_start);

        if args.test_source is not None:
            local_pp = log_likelihood(model, test_source_corpus, test_target_corpus)
            pp += local_pp #     pp += local_pp;

        pp = math.exp(-pp/num_test_instances) #   pp = exp(-pp/num_test_instances);
        sys.stderr.write(11*" "+str(iteration_time)+10*" "+str(av_f/num_training_instances)) #   cerr << setw(11) << iteration_time << setw(10) << av_f/num_training_instances;
        if args.test_source is not None: #   if (vm.count("test-source")) {
            sys.stderr.write(" "*13+str(pp)) #     cerr << setw(13) << pp;
        sys.stderr.write(" |\n") #   cerr << " |" << endl;
     # }


     # if (vm.count("model-out")) {
     #   cout << "Writing trained model to " << vm["model-out"].as<string>() << endl;
     #   std::ofstream f(vm["model-out"].as<string>().c_str());
     #   boost::archive::text_oarchive ar(f);
     #   ar << model;
     # }

def __get_args():
    parser = argparse.ArgumentParser(description='Training for neural translation models: Copyright 2014 Ed Grefenstette.')
    parser.add_argument("-c","--config", type=str,help="config file specifying additional command line options")
    parser.add_argument("-s","--source", help="corpus of sentences, one per line")
    parser.add_argument("-t","--target", type=str, help="corpus of sentences, one per line")
    parser.add_argument("--test-source", type=str, help="corpus of test sentences to be evaluated at each iteration")
    parser.add_argument("--test-target", type=str, help="corpus of test sentences to be evaluated at each iteration")
    parser.add_argument("--iterations", type=int, default=10, help="number of passes through the data")
    parser.add_argument("--minibatch-size", type=int, default=100, help="number of sentences per minibatch")
    parser.add_argument("--instances", type=int, default=sys.maxint, help="training instances per iteration")
    parser.add_argument("-n","--order", type=int, default=3, help="ngram order")
    parser.add_argument("-m","--model-in", type=str, help="initial model")
    parser.add_argument("-o","--model-out", type=str, default="model", help="base filename of model output files")
    parser.add_argument("-r","--l2", type=float, default=0.0, help="l2 regularisation strength parameter")
    parser.add_argument("--l1", type=float, default=0.0, help="l1 regularisation strength parameter")
    parser.add_argument("--source-l2", type=float, help="source regularisation strength parameter")
    parser.add_argument("--dump-frequency", type=int, default=0, help="dump model every n minibatches.")
    parser.add_argument("--word-width", type=int, default=100, help="Width of word representation vectors.")
    parser.add_argument("--threads", type=int, default=1, help="number of worker threads.")
    parser.add_argument("--step-size", type=float, default=1.0, help="SGD batch stepsize, it is normalised by the number of minibatches.")
    parser.add_argument("--classes", type=int, default=100, help="number of classes for factored output.")
    parser.add_argument("--class-file", type=str, help="file containing word to class mappings in the format: <class> <word> <frequence>.")
    parser.add_argument("--window", type=int, default=-1, help="Width of window of source words conditioned on.")
    parser.add_argument("--no-source-eos", help="do not add end of sentence tag to source representations.")
    parser.add_argument("--replace-source-dict", help="replace the source dictionary of a loaded model with a new one.")
    parser.add_argument("-v","--verbose", default=False, action="store_true", help="print perplexity for each sentence (1) or input token (2) ")
    parser.add_argument("--randomise", default=False, action="store_true", help="visit the training tokens in random order.")
    parser.add_argument("--diagonal-contexts", default=False, action="store_true", help="Use diagonal context matrices (usually faster).")
    parser.add_argument("--non-linear", default=False, action="store_true", help="use a non-linear hidden layer.")

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("-c","--config", type=str)
    config_args, _ = config_parser.parse_known_args()

    if config_args.config is not None:
        config_file = StringIO.StringIO()
        config_file.write('[dummysection]\n')
        config_file.write(open('myrealconfig.ini').read())
        config_file.seek(0, os.SEEK_SET)
        cp = ConfigParser.ConfigParser()
        cp.readfp(config_file)
        parser.set_defaults(**dict(cp.items('dummysection')))

    args = parser.parse_args()
    return args


def __get_configuration(args):
    configuration = config.ModelData() # ModelData config;
    configuration.l1_parameter = float(args.l1) # config.l1_parameter = vm["l1"].as<float>();
    configuration.l2_parameter = float(args.l2) # config.l2_parameter = vm["l2"].as<float>();
    if args.source_l2 is not None: #if (vm.count("source-l2"))
        configuration.source_l2_parameter = float(args.source_l2) #   config.source_l2_parameter = vm["source-l2"].as<float>();
    else:
        configuration.source_l2_parameter = configuration.l2_parameter #   config.source_l2_parameter = config.l2_parameter;
    configuration.word_representation_size = int(args.word_width) # config.word_representation_size = vm["word-width"].as<int>();
    configuration.threads = int(args.threads) # config.threads = vm["threads"].as<int>();
    configuration.ngram_order = int(args.order) # config.ngram_order = vm["order"].as<int>();
    configuration.verbose = args.verbose # config.verbose = vm.count("verbose");
    configuration.classes = int(args.classes) # config.classes = vm["classes"].as<int>();
    configuration.diagonal = args.diagonal_contexts # config.diagonal = vm.count("diagonal-contexts");
    configuration.nonlinear = args.non_linear # config.nonlinear = vm.count("non-linear");
    configuration.source_window_width = int(args.window) # config.source_window_width = vm["window"].as<int>();
    configuration.source_eos = not args.no_source_eos # config.source_eos = !vm.count("no-source-eos");

    return configuration


def _freq_bin_type(corpus, num_classes, classes, d, class_bias):
    if type(corpus) is not str:
        raise TypeError("Expecting corpus to be of type str. Type was: %s." % type(corpus).__name__)
    if type(num_classes) is not int:
        raise TypeError("Expecting num_classes to be of type int. Type was: %s." % type(num_classes).__name__)
    if type(classes) is not list:
        raise TypeError("Expecting classes to be of type list. Type was: %s." % type(classes).__name__)
    if type(d) is not corpus.Dict:
        raise TypeError("Expecting d to be of type Dict. Type was: %s." % type(d).__name__)
    if type(class_bias) is not np.ndarray:
        raise TypeError("Expecting class_bias to be of type ndarray. Type was: %s." % type(class_bias).__name__)

    tmp_dict = {}
    counts = []
    sum = 0
    eos_sum = 0
    eos = "</s>"

    with open(corpus, 'r') as f:
        for line in f:
            for token in line.strip().split():
                if token == eos:
                    continue
                w_id = len(tmp_dict)
                tmp_dict[token] = w_id
                assert w_id <= len(counts)
                if w_id == len(counts):
                    counts.append((token, 1))
                else:
                    tok, cnt = counts[w_id]
                    counts[w_id] = (tok, cnt + 1 )
                sum += 1

            eos_sum += 1

    counts.sort(key=operator.itemgetter(1), reverse=True)

    if len(classes) > 0:
        while len(classes) > 0: classes.pop()
    classes.append(0)
    classes.append(2)
    class_bias[0] = math.log(eos_sum)
    bin_size = sum // (num_classes - 1)

    mass = 0
    for i in range(len(counts)):
        id = d.Convert(counts[0][0])

        mass += counts[i][1]
        if mass > bin_size:
            sum -= mass
            bin_size = (sum) // (num_classes - len(classes))
            class_bias[len(classes) - 1] = math.log(mass)

            classes.append(id + 1)

            mass = 0
    if classes[-1] != d.size():
        classes.append(d.size())

    class_bias -= math.log(eos_sum + sum)

    sys.stderr.write("Binned %d types in %d classes with an average of %f types per bin.\n" % (d.size(), len(classes) - 1, float(d.size()) / float(len(classes) - 1)))


def _classes_from_file(class_file, classes, d):
    if type(class_file) is not str:
        raise TypeError("Expecting class_file to be of type str. Type was: %s." % type(class_file).__name__)
    if type(classes) is not list:
        raise TypeError("Expecting classes to be of type list. Type was: %s." % type(classes).__name__)
    if type(d) is not corpus.Dict:
        raise TypeError("Expecting d to be of type Dict. Type was: %s." % type(d).__name__)

    class_freqs = []
    if len(classes) > 0:
        while len(classes) > 0: classes.pop()
    classes.append(0)
    classes.append(2)
    mass = 0
    total_mass = 0
    prev_class_str = ""
    class_str = ""
    token_str = ""
    freq_str = ""
    with open(class_file, 'r') as f:

        for line in f:
            class_str, token_str, freq_str = line.strip().split()
            w_id = d.Convert(token_str)
            if len(prev_class_str) > 0 and class_str != prev_class_str:
                class_freqs.append(math.log(mass))
                classes.append(w_id)#     classes.push_back(w_id);
                mass = 0
            freq = int(freq_str)
            mass += freq
            total_mass += freq
            prev_class_str = class_str

        class_freqs.append(math.log(mass))
        classes.append(d.size())
        class_bias = np.zeros((len(class_freqs), 1))
        for i in range(len(class_freqs)):
            class_bias[i] = class_freqs[i] - math.log(total_mass)
            sys.stderr.write("Read %d types in %d classes with an average of %f types per bin.\n" % (d.size(),
                                                                                                     len(classes) - 1,
                                                                                                     float(d.size())/float(len(classes) - 1)))
        return class_bias


def load_classes(args, target_dict, configuration):
    classes = []
    if args.class_file is not None: # if (vm.count("class-file")) {
        sys.stderr.write("--class-file set, ignoring --classes.\n")
        class_bias = _classes_from_file(args.class_file, classes, target_dict)
        configuration.classes = len(classes) - 1
    else:
        class_bias = np.zeros((configuration.classes, 1))
        _freq_bin_type(args.target, configuration.classes, classes, target_dict, class_bias)
    return classes, class_bias


def load_model(args, config, source_dict, target_dict, classes):
    Model = cnlm.CNLMBase # TODO: Update this with a better model class.
    model = Model(config, source_dict, target_dict, classes) # AdditiveCNLM model(config, source_dict, target_dict, classes);
    frozen_model = False # bool frozen_model = false;
    replace_source_dict = False # bool replace_source_dict = false;
    if args.replace_source_dict is not None: # if (vm.count("replace-source-dict")) {
        assert args.model_in is not None #   assert(vm.count("model-in"));
        replace_source_dict = True #   replace_source_dict = true;

    if args.model_in: # if (vm.count("model-in")) {
        raise NotImplementedError("Need to implement model loading code here.") # INSERT MODEL LOADING CODE HERE
         #   std::ifstream f(vm["model-in"].as<string>().c_str());
         #   boost::archive::text_iarchive ar(f);
         #   ar >> model;
         #   target_dict = model.label_set();
         #   if(!replace_source_dict)
         #     source_dict = model.source_label_set();
         #   // Set dictionary update to false and freeze model parameters in general.
         #   frozen_model = true;
         #   // Adjust config.update parameter, as this is dependent on the specific
         #   // training run and not on the model per se.
         #   model.config.updates = config.updates;
    return model, frozen_model, replace_source_dict


def read_training_sentences(args, target_dict, frozen_model, end_id, target_corpus):
    num_training_instances = 0
    with open(args.target, 'r') as f:
        for line in f:
            s = []
            for token in line.strip().split():
                w = target_dict.Convert(token, frozen_model)
                if (w < 0):
                    sys.stderr.write("%s %d\n" % (token, w))
                    assert False, "Word found in training target corpus, which wasn't encountered in originally trained and loaded model."
                s.append(w)

            s.append(end_id)
            num_training_instances += len(s)
            target_corpus.append(s)
    return num_training_instances


def read_source_sentences(args, source_dict, frozen_model, replace_source_dict, configuration, end_id, source_corpus):
    with open(args.source, 'r') as f:
        for line in f:
            s = []
            for token in line.strip().split():
                w = source_dict.Convert(token, frozen_model and not replace_source_dict)
                if (w < 0):
                    sys.stderr.write("%s %d\n" % (token, w))
                    assert False, "Word found in training source corpus, which wasn't encountered in originally trained and loaded model."
                s.append(w)
            if configuration.source_eos:
                s.append(end_id)
            source_corpus.append(s)


def read_test_sentences(args, source_dict, target_dict, configuration, end_id, test_source_corpus, test_target_corpus):
    with open(args.test_source, 'r') as f: #   ifstream test_source_in(vm["test-source"].as<string>().c_str());
        for line in f: #   while (getline(test_source_in, line)) {
            s = [] #     Sentence& s = test_source_corpus.back();
            for token in line.strip().split(): #     while (line_stream >> token) {
                w = source_dict.Convert(token, True) #       WordId w = source_dict.Convert(token, true);
                if w < 0: #       if (w < 0) {
                    sys.stderr.write("%s %d\n" % (token, w))
                    assert False, "Unknown word found in test source corpus." #         cerr << token << " " << w << endl;
                 #       }
                s.append(w) #       s.push_back(w);
             #     }
            if configuration.source_eos: #     if (config.source_eos)
                s.append(end_id) #       s.push_back(end_id);
            test_source_corpus.append(s)

    num_test_instances = 0
    with open(args.test_target, 'r') as f: #   ifstream test_target_in(vm["test-target"].as<string>().c_str());
        for line in f: #   while (getline(test_target_in, line)) {
            s = [] #     Sentence& s = test_target_corpus.back();
            for token in line.strip().split(): #     while (line_stream >> token) {
                w= target_dict.Convert(token, True) #       WordId w = target_dict.Convert(token, true);
                if w < 0: #       if (w < 0) {
                    sys.stderr.write("%s %d\n" % (token, w))
                    assert False, "Unknown word found in test target corpus." #         cerr << token << " " << w << endl;
                s.append(w) #       s.push_back(w);
             #     }
            s.append(end_id) #     s.push_back(end_id);
            test_target_corpus.append(s)
            num_test_instances += s.size() #     num_test_instances += s.size();
    return num_test_instances


def reinitialise_model(frozen_model, replace_source_dict, source_corpus, target_corpus, model):
    if not frozen_model: # if(!frozen_model) {
        raise NotImplementedError #   model.reinitialize(config, source_dict, target_dict, classes);
        raise NotImplementedError #   cerr << "(Re)initializing model based on training data." << endl;

    elif replace_source_dict: # else if(replace_source_dict) {
        raise NotImplementedError #   model.expandSource(source_dict);
        raise NotImplementedError #   cerr << "Replacing source dictionary based on training data." << endl;

    if not frozen_model: # if(!frozen_model)
        raise NotImplementedError #   model.FB = class_bias;

    for s in range(len(source_corpus)): # for (size_t s = 0; s < source_corpus.size(); ++s)
        raise NotImplementedError #   model.length_ratio += (Real(source_corpus.at(s).size()) / Real(target_corpus.at(s).size()));

    raise NotImplementedError # model.length_ratio /= Real(source_corpus.size());

    training_indices = [0]*len(target_corpus)
    unigram = np.zeros((model.labels(),1))
    for i in range(len(training_indices)):
        for j in range(len(target_corpus)):
            unigram[target_corpus[i][j]] += 1
        training_indices[i] = i

    if not frozen_model:
        model.B = np.log((unigram + 1.0)/(unigram.sum() + unigram.shape))


def log_likelihood(model, test_source_corpus, test_source):
    raise NotImplementedError

if __name__ == '__main__':
    main()
