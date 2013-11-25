import sys
import argparse
import os
import shlex
from tempfile import gettempdir

from copy import deepcopy as copy
from collections import OrderedDict
from subprocess import Popen, PIPE

from yaml import load as yload
from yaml import dump as ydump

try:
    from yaml import CLoader as YLoader
    from yaml import CDumper as YDumper
except ImportError:
    from yaml import Loader as YLoader
    from yaml import Dumper as YDumper

REVISION = "$Rev: 6 $"

class DefaultHelpParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n\n' % message)
        self.print_help()
        sys.exit(2)

class DefaultActionObject(object):
    """docstring for ActionObject"""

    def __init__(self, argv):

        self.actionObjects = [ActionReadyTrainingData(self),
                              ActionTrainModel(self),
                              ActionPreprocessTestset(self),
                              ActionEvaluateModel(self)]

        self.parser = self.get_parser()
        for action in self.actionObjects:
            action.extend_parser()
        self.args = self.get_args(argv)

        self.command = self.args.command

        if self.args.verbose:
            self.vwrite("Run CNLM label experiments. Copyright 2013 Ed Grefenstette, %s.", REVISION)

        self.config = {}

        if self.args.save_config_path:
            self.save()

        if not self.valid():
            self.parser.error("No command specified.")

        must_ready = 'r' in self.command
        must_train = 't' in self.command
        must_preprocess = 'p' in self.command
        must_evaluate = 'e' in self.command

        self.actions = OrderedDict()

        if must_ready:
            self.actions['r'] = ActionReadyTrainingData(self)
        if must_train:
            self.actions['t'] = ActionTrainModel(self)
        if must_preprocess:
            self.actions['p'] = ActionPreprocessTestset(self)
        if must_evaluate:
            self.actions['e'] = ActionEvaluateModel(self)

        for actionkey in self.actions:
            self.actions[actionkey].initialise_config()

        if self.args.config:
            self.load_config()

        for actionkey in self.actions:
            action = self.actions[actionkey]
            action.process_args()
            if not action.valid():
                self.parser.error("Not enough arguments.")

        if must_ready:
            raise NotImplementedError("Readying dataset not implemented yet")

        if must_train:
            raise NotImplementedError("Training not implemented yet")

        if must_preprocess:
            self.actions['p'].do_action()

        if must_evaluate:
            raise NotImplementedError("Evaluation not implemented yet")

    def get_parser(self):
        parser = DefaultHelpParser(description='Run CNLM label experiments.', epilog="Copyright 2013 Ed Grefenstette, %s" % REVISION)
        parser.add_argument('-c', '--config', help="Load configuration file CONFIG.", default=None)
        parser.add_argument('--save-config', dest="save_config_path", metavar="CONFIGPATH", help="Save configuration and quit.", type=str, default=None)
        parser.add_argument('-v', '--verbose', default=False, action='store_true', help="Verbose output to stderr.")

        parser.add_argument('command', metavar="COMMAND", default=None, help="Command(s) to be executed. One or more of t(rain model), p(rocess test set), e(valuate model).", nargs="?")
        return parser

    def get_args(self, argv):
        return self.parser.parse_args(argv)

    @classmethod
    def ewrite(self, message, *args):
        if not message.endswith('\n'):
            message = message+"\n"
        sys.stderr.write(message % tuple(args))

    @classmethod
    def vwrite(self, message, *args):
        message = "VERBOSE: "+message
        self.ewrite(message, *args)

    def initialise_config(self):
        self.config["group_exec"] = {}
        self.config["group_exec"]["oxcg_bin"] = None  # Mandatory

    def load_config(self):
        ## Load configuration if specified

        existing_config = yload(open(self.args.config), YLoader)

        for groupkey in existing_config:
            for key in existing_config[groupkey]:
                self.config[groupkey][key] = existing_config[groupkey][key]

    def valid(self):

        ## Validate args
        throw_error = False

        if not self.command:
            return throw_error
        else:
            return not throw_error

    def config_clean_copy(self, cleanconfig=None):
        if cleanconfig is None:
            cleanconfig = copy(self.config)

        if not cleanconfig["group_exec"]["oxcg_bin"]: del cleanconfig["group_exec"]["oxcg_bin"]
        if not cleanconfig["group_exec"]: del cleanconfig["group_exec"]
        return cleanconfig

    def save(self):
        configfile = open(self.args.save_config_path, 'w')

        for action in self.actionObjects:
            action.initialise_config()

        if self.args.config:
            self.load_config()

        for action in self.actionObjects:
            action.process_args()

        self.save_config(configfile)
        configfile.close()

        if self.args.verbose:
            self.vwrite("Saved configuration to %s and quitting.", self.args.save_config_path)
            sys.exit(0)

    def save_config(self, configfile):
        cleanconfig = self.config_clean_copy(self.config)
        ydump(cleanconfig, configfile, YDumper, default_flow_style=False)

    def print_config(self):
        raise NotImplementedError


class ActionReadyTrainingData(DefaultActionObject):
    """docstring for ActionTrainModel"""
    def __init__(self, that):
        self.that = that

    def extend_parser(self):
        that=self.that

    def initialise_config(self):
        that=self.that

    def process_args(self):
        that=self.that

    def valid(self):
        that=self.that
        throw_error = False

        raise NotImplementedError
        return not throw_error

    def config_clean_copy(self, cleanconfig):
        that=self.that
        raise NotImplementedError

class ActionTrainModel(DefaultActionObject):
    """docstring for ActionTrainModel"""
    def __init__(self, that):
        self.that = that

    def extend_parser(self):
        that=self.that
        group_exec = that.parser.add_argument_group("executable/script paths")
        group_exec.add_argument('--oxcg-bin', dest='oxcg_bin', help="Path for oxcg/bin directory.", default=None)

        group_train = that.parser.add_argument_group("Training parameters")

    def initialise_config(self):
        that=self.that
        that.config["group_exec"] = {}
        that.config["group_exec"]["oxcg_bin"] = None  # Mandatory

    def process_args(self):
        that=self.that
        if that.args.oxcg_bin:
            that.config["group_exec"]["oxcg_bin"] = that.args.oxcg_bin

    def valid(self):
        that=self.that
        throw_error = False
        if not that.config["group_exec"]["oxcg_bin"]: throw_error = True

        raise NotImplementedError
        return not throw_error

    def config_clean_copy(self, cleanconfig):
        that=self.that
        raise NotImplementedError

class ActionPreprocessTestset(DefaultActionObject):
    """docstring for ActionPreprocessTestset"""
    def __init__(self, that):
        self.that = that

    def extend_parser(self):
        that=self.that
        group_testset_reformat = that.parser.add_argument_group("test set reformatting arguments")

        group_testset_reformat.add_argument('--test-sentences', dest='test_sentences', type=str,
                                            help='Test sentences (one per line).', default=None)
        group_testset_reformat.add_argument('--output-source', dest='output_source', type=str, default=None,
                                             help='Output source file.')
        group_testset_reformat.add_argument('--output-target', dest='output_target', type=str, default=None,
                                            help='Output target file.')
        group_testset_reformat.add_argument('-l', '--labels', dest='labels', default=None,
                                            help='List of comma-separated labels, of the form "label1,label2,...,labeln".')
        group_testset_reformat.add_argument('-r', '--repetitions', dest='repetitions', default=None, type=int,
                                            help='(OPTIONAL) Number of times the label lines are repeated (advanced feature).')
        group_testset_reformat.add_argument('-d', '--dynamic-repetitions', dest='dynamic_repetitions', default=None, action='store_true',
                                            help='(OPTIONAL) Base number of times label lines are repeated on the length of the source sentence (advanced feature).')
        group_testset_reformat.add_argument('--data-dict-output', dest="datadict_output", matavar="DATADICT", default=None, help="Output file for data dictionary.")

    def initialise_config(self):
        that=self.that
        that.config["group_testset_reformat"] = {}
        that.config["group_testset_reformat"]["test_sentences"] = None           # Mandatory
        that.config["group_testset_reformat"]["output_source"] = None            # Mandatory
        that.config["group_testset_reformat"]["output_target"] = None            # Mandatory
        that.config["group_testset_reformat"]["labels"] = None                   # Mandatory
        that.config["group_testset_reformat"]["datadict"] = None                 # Mandatory
        that.config["group_testset_reformat"]["repetitions"] = 1                 # Optional
        that.config["group_testset_reformat"]["dynamic_repetitions"] = False     # Optional


    def process_args(self):
        that=self.that

        ## Override config with command line args, if non-null
        if that.args.test_sentences:
            that.config["group_testset_reformat"]["test_sentences"] = that.args.test_sentences
        if that.args.output_source:
            that.config["group_testset_reformat"]["output_source"] = that.args.output_source
        if that.args.output_target:
            that.config["group_testset_reformat"]["output_target"] = that.args.output_target
        if that.args.labels:
            that.config["group_testset_reformat"]["labels"] = that.args.labels
        if that.args.repetitions:
            that.config["group_testset_reformat"]["repetitions"] = that.args.repetitions
        if that.args.dynamic_repetitions:
            that.config["group_testset_reformat"]["dynamic_repetitions"] = that.args.dynamic_repetitions
        if that.args.datadict_output:
            that.config["group_testset_reformat"]["datadict"] = that.args.datadict_output

    def valid(self):
        that=self.that
        throw_error = False
        if not that.config["group_testset_reformat"]["test_sentences"]: throw_error = True
        if not that.config["group_testset_reformat"]["output_source"]: throw_error = True
        if not that.config["group_testset_reformat"]["output_target"]: throw_error = True
        if not that.config["group_testset_reformat"]["labels"]: throw_error = True
        if not that.config["group_testset_reformat"]["datadict"]: throw_error = True
        return not throw_error

    def config_clean_copy(self, cleanconfig):
        that=self.that
        if not that.config["group_testset_reformat"]["test_sentences"]:      del cleanconfig["group_testset_reformat"]["test_sentences"]
        if not that.config["group_testset_reformat"]["output_source"]:       del cleanconfig["group_testset_reformat"]["output_source"]
        if not that.config["group_testset_reformat"]["output_target"]:       del cleanconfig["group_testset_reformat"]["output_target"]
        if not that.config["group_testset_reformat"]["labels"]:              del cleanconfig["group_testset_reformat"]["labels"]
        if that.config["group_testset_reformat"]["repetitions"] is 1:        del cleanconfig["group_testset_reformat"]["repetitions"]
        if not that.config["group_testset_reformat"]["dynamic_repetitions"]: del cleanconfig["group_testset_reformat"]["dynamic_repetitions"]
        if not that.config["group_testset_reformat"]["datadict"]:            del cleanconfig["group_testset_reformat"]["datadict"]
        if not that.config["group_testset_reformat"]:                        del cleanconfig["group_testset_reformat"]
        return cleanconfig

    def process_testset(self, labels, label_repetitions, istream, ostream_source, ostream_target, dynamic_repetitions):

        num_labels = len(labels)

        test_sentences = [line.strip() for line in istream if len(line.strip()) > 0 and not line.strip().startswith('#')]
        num_test_sentences = len(test_sentences)

        datadict = {}

        for k in range(num_test_sentences):
            sentence = test_sentences[k]
            sdict = datadict.setdefault(sentence,{})
            for i in range(num_labels):
                theta_k_i = 's(%d,%d)' % (k, i)
                label = labels[i]
                sdict[theta_k_i] = label
                ostream_source.write('%s\n' % theta_k_i)
                ostream_target.write('%s\n' % sentence)
                if dynamic_repetitions:
                    label_repetitions = len(test_sentences[k].split())
                for n in range(label_repetitions):
                    ostream_source.write('%s\n' % theta_k_i)
                    ostream_target.write('%s\n' % label)

        return datadict

    def do_action(self):
        that=self.that
        if that.args.verbose:
            that.vwrite("Preprocessing test script %s.", that.config["group_testset_reformat"]["test_sentences"])

        ppconfig = that.config["group_testset_reformat"]
        istream = open(ppconfig["test_sentences"])
        ostream_source = open(ppconfig["output_source"], 'w')
        ostream_target = open(ppconfig["output_target"], 'w')
        datadict = self.process_testset(labels=ppconfig["labels"],
                                        label_repetitions=ppconfig["repetitions"],
                                        istream=istream,
                                        ostream_source=ostream_source,
                                        ostream_target=ostream_target,
                                        dynamic_repetitions=ppconfig["dynamic_repetitions"])
        istream.close()
        ostream_source.close()
        ostream_target.close()

        ddictstream = open(ppconfig['datadict'], 'w')
        ydump(datadict, ddictstream, YDumper, default_flow_style=False)
        ddictstream.close()

        if that.args.verbose:
            that.vwrite("Done preprocessing test script.")
            that.vwrite("Output written to %s and %s.", ppconfig["output_source"], ppconfig["output_target"])

class ActionEvaluateModel(DefaultActionObject):
    """docstring for ActionTrainModel"""
    def __init__(self, that):
        self.that = that

    def extend_parser(self):
        that=self.that
        group_eval = that.parser.add_argument_group("evaluation arguments")
        group_eval.add_argument("--eval-thetas", dest="evaluation_thetas", default=None, help="Theta labels for test set.")
        group_eval.add_argument("--eval-targets", dest="evaluation_targets", default=None, help="Test sentences and label enumerations.")
        group_eval.add_argument("--data-dict", dest="datadict", default=None, help="Input file for data dictionary.")
        group_eval.add_argument("--candidate", dest="input_model", default=None, help="Path to model to be evaluated.")

    def initialise_config(self):
        that=self.that
        that.config["group_exec"] = {}
        that.config["group_exec"]["oxcg_bin"] = None  # Mandatory

        that.config["group_eval"] = {}
        that.config["group_eval"]["evaluation_thetas"] = None # Mandatory
        that.config["group_eval"]["evaluation_targets"] = None # Mandatory
        that.config["group_eval"]["datadict"] = None # Mandatory
        that.config["group_eval"]["input_model"] = None # Mandatory

    def process_args(self):
        that=self.that
        if that.args.oxcg_bin:
            that.config["group_exec"]["oxcg_bin"] = that.args.oxcg_bin

        if that.args.evaluation_thetas:
            that.config["group_eval"]["evaluation_thetas"] = that.args.evaluation_thetas
        if that.args.evaluation_targets:
            that.config["group_eval"]["evaluation_targets"] = that.args.evaluation_targets
        if that.args.evaluation.datadict:
            that.config["group_eval"]["datadict"] = that.args.datadict
        if that.args.input_model:
            that.config["group_eval"]["input_model"] = that.args.input_model

    def valid(self):
        that=self.that
        throw_error = False
        if not that.config["group_exec"]["oxcg_bin"]: throw_error = True

        if not that.config["group_eval"]["evaluation_thetas"]: throw_error = True
        if not that.config["group_eval"]["evaluation_targets"]: throw_error = True
        if not that.config["group_eval"]["datadict"]: throw_error = True
        if not that.config["group_eval"]["input_model"]: throw_error = True
        return not throw_error

    def config_clean_copy(self, cleanconfig):
        that=self.that
        raise NotImplementedError

    def do_action(self):
        that=self.that
        self.theta_source = that.config["group_eval"]["evaluation_thetas"]
        self.joint_target = that.config["group_eval"]["evaluation_targets"]
        self.datadict = yload(open(that.config["group_eval"]["datadict"]), YLoader)
        self.modelpath = that.config["group_eval"]["input_model"]
        self.mapmodelpath = os.path.join(gettempdir(), "mapmodel")

    def learn_map_thetas(self):
        that=self.that
        train_binary = os.path.join(that.config["group_exec"]["oxcg_bin"],"train_cnlm")

        source_and_target_args = ["-s", self.theta_source,
                                  "-t", self.joint_target,
                                  "-m", self.modelpath,
                                  "-o", self.mapmodelpath]

        learning_args = ["--iterations", str(iterations),
                         "--minibatch-size", str(mbsize),
                         "--l2", str(lambda_value),
                         "--step-size", str(stepsize),
                         "--class-file", classfilepath,
                         "--no-source-eos=true",
                         "--replace-source-dict=true"]

        freeze_weights_args=["--updateT=false",
                             "--updateS=true",
                             "--updateC=false",
                             "--updateR=false",
                             "--updateQ=false",
                             "--updateF=false",
                             "--updateFB=false",
                             "--updateB=false"]

        args = [train_binary] + source_and_target_args +learning_args + freeze_weights_args

        map_estimate_process = Popen(args, stdout=PIPE, stderr=PIPE)
        output, error = map_estimate_process.communicate()

    def get_estimates(self):
        that=self.that

        source_lines = [line.strip() for line in open(self.theta_source)]
        target_lines = [line.strip() for line in open(self.joint_target)]

        perplexity_binary = os.path.join(that.config["group_exec"]["oxcg_bin"],"perplexity")
        args = [perplexity_binary, "-m", self.mapmodelpath, "-s", self.theta_source, "-t", self.joint_target, "--print-sentence-llh"]

        cond_probs_process = Popen(args, stdout=PIPE, stderr=PIPE)
        cond_probs, _ = cond_probs_process.communicate()

        score_tuples = zip(source_lines, target_lines, cond_probs)

        return score_tuples


def run_evaluation():
    pass

def main(argv):
    mainAction = DefaultActionObject(argv)

if __name__ == '__main__':
    main(sys.argv[1:])