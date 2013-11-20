import sys
import argparse
# import subprocess
# import shlex

from copy import deepcopy as copy

from yaml import load as yload
from yaml import dump as ydump

try:
    from yaml import CLoader as YLoader
    from yaml import CDumper as YDumper
except ImportError:
    from yaml import Loader as YLoader
    from yaml import Dumper as YDumper

REVISION = "$Rev: 3 $"

class DefaultHelpParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n\n' % message)
        self.print_help()
        sys.exit(2)

class DefaultActionObject(object):
    """docstring for ActionObject"""
    def __init__(self, argv):

        parser, args = self.parse(argv)

        self.command = args.command

        if args.verbose:
            self.vwrite("Run CNLM label experiments. Copyright 2013 Ed Grefenstette, %s.", REVISION)

        self.config = {}

        if args.save_config_path:
            configfile = open(args.save_config_path, 'w')

            self.save_config(configfile)
            configfile.close()

            if args.verbose:
                self.vwrite("Saved configuration to %s and quitting.", args.save_config_path)
                sys.exit(0)

        if not self.valid():
            parser.error("No command specified.")

        must_ready = 'r' in self.command
        must_train = 't' in self.command
        must_preprocess = 'p' in self.command
        must_evaluate = 'e' in self.command

        self.actions = []

        if must_ready: self.actions.append(ActionReadyTrainingData(self))
        if must_train: self.actions.append(ActionTrainModel(self))
        if must_preprocess: self.actions.append(ActionPreprocessTestset(self))
        if must_evaluate: self.actions.append(ActionEvaluateModel(self))

        for action in self.actions:
            action.initialise_config()
            action.process_args(args)
            if not action.valid():
                parser.error("Not enough arguments.")

        if must_ready:
            raise NotImplementedError("Readying dataset not implemented yet")

        if must_train:
            raise NotImplementedError("Training not implemented yet")

        if must_preprocess:
            if args.verbose:
                self.vwrite("Preprocessing test script %s.", self.config["group_testset_reformat"]["test_sentences"])

            ppconfig = self.config["group_testset_reformat"]
            istream = open(ppconfig["test_sentences"])
            ostream_source = open(ppconfig["output_source"], 'w')
            ostream_target = open(ppconfig["output_target"], 'w')
            self.process_testset(labels=ppconfig["labels"],
                                 label_repetitions=ppconfig["repetitions"],
                                 istream=istream,
                                 ostream_source=ostream_source,
                                 ostream_target=ostream_target,
                                 dynamic_repetitions=ppconfig["dynamic_repetitions"])
            istream.close()
            ostream_source.close()
            ostream_target.close()

            if args.verbose:
                self.vwrite("Done preprocessing test script.")
                self.vwrite("Output written to %s and %s.", ppconfig["output_source"], ppconfig["output_target"])

        if must_evaluate:
            raise NotImplementedError("Evaluation not implemented yet")

    @staticmethod
    def parse(argv):
        parser = DefaultHelpParser(description='Run CNLM label experiments.', epilog="Copyright 2013 Ed Grefenstette, %s" % REVISION)
        parser.add_argument('-c', '--config', help="Load configuration file CONFIG.", default=None)
        parser.add_argument('--save-config', dest="save_config_path", metavar="CONFIGPATH", help="Save configuration and quit.", type=str, default=None)
        parser.add_argument('-v', '--verbose', default=False, action='store_true', help="Verbose output to stderr.")

        group_exec = parser.add_argument_group("executable/script paths")

        # group_exec.add_argument('--python-scripts', dest="python_script_path", help="Path for python helper scripts.", default=None)
        group_exec.add_argument('--oxcg-bin', dest='oxcg_bin', help="Path for oxcg/bin directory.", default=None)

        group_testset_reformat = parser.add_argument_group("test set reformatting arguments")

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

        parser.add_argument('command', metavar="COMMAND", default=None, help="Command(s) to be executed. One or more of t(rain model), p(rocess test set), e(valuate model).", nargs="?")

        args = parser.parse_args(argv)

        return parser, args

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

    def load_config(self, args):
        ## Load configuration if specified

        existing_config = yload(open(args.config), YLoader)

        for groupkey in existing_config:
            for key in existing_config[groupkey]:
                self.config[groupkey][key] = existing_config[groupkey][key]

    def process_args(self, args):
        ## Override config with command line args, if non-null

        if args.oxcg_bin:
            self.config["group_exec"]["oxcg_bin"] = args.oxcg_bin

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

    def save_config(self, configfile):
        cleanconfig = self.config_clean_copy(self.config)
        ydump(cleanconfig, configfile, YDumper, default_flow_style=False)

    def print_config(self):
        raise NotImplementedError


class ActionReadyTrainingData(DefaultActionObject):
    """docstring for ActionTrainModel"""
    def __init__(self, that):
        self.that = that

    def initialise_config(self):
        that=self.that
        raise NotImplementedError

    def process_args(self, args):
        that=self.that
        raise NotImplementedError

    def valid(self):
        that=self.that
        throw_error = False

        raise NotImplementedError
        return not throw_error

    def config_clean_copy(self, cleanconfig):
        that=self.that
        raise NotImplementedError

class ActionPreprocessTestset(DefaultActionObject):
    """docstring for ActionPreprocessTestset"""
    def __init__(self, that):
        self.that = that

    def initialise_config(self):
        that=self.that
        that.config["group_testset_reformat"] = {}
        that.config["group_testset_reformat"]["test_sentences"] = None           # Mandatory
        that.config["group_testset_reformat"]["output_source"] = None            # Mandatory
        that.config["group_testset_reformat"]["output_target"] = None            # Mandatory
        that.config["group_testset_reformat"]["labels"] = None                   # Mandatory
        that.config["group_testset_reformat"]["repetitions"] = 1                 # Optional
        that.config["group_testset_reformat"]["dynamic_repetitions"] = False     # Optional


    def process_args(self, args):
        that=self.that

        ## Override config with command line args, if non-null
        if args.test_sentences:
            that.config["group_testset_reformat"]["test_sentences"] = args.test_sentences
        if args.output_source:
            that.config["group_testset_reformat"]["output_source"] = args.output_source
        if args.output_target:
            that.config["group_testset_reformat"]["output_target"] = args.output_target
        if args.labels:
            that.config["group_testset_reformat"]["labels"] = args.labels
        if args.repetitions:
            that.config["group_testset_reformat"]["repetitions"] = args.repetitions
        if args.dynamic_repetitions:
            that.config["group_testset_reformat"]["dynamic_repetitions"] = args.dynamic_repetitions

    def valid(self):
        that=self.that
        throw_error = False
        if not that.config["group_testset_reformat"]["test_sentences"]: throw_error = True
        if not that.config["group_testset_reformat"]["output_source"]: throw_error = True
        if not that.config["group_testset_reformat"]["output_target"]: throw_error = True
        if not that.config["group_testset_reformat"]["labels"]: throw_error = True
        return not throw_error

    def config_clean_copy(self, cleanconfig):
        that=self.that
        if not that.config["group_testset_reformat"]["test_sentences"]:      del cleanconfig["group_testset_reformat"]["test_sentences"]
        if not that.config["group_testset_reformat"]["output_source"]:       del cleanconfig["group_testset_reformat"]["output_source"]
        if not that.config["group_testset_reformat"]["output_target"]:       del cleanconfig["group_testset_reformat"]["output_target"]
        if not that.config["group_testset_reformat"]["labels"]:              del cleanconfig["group_testset_reformat"]["labels"]
        if that.config["group_testset_reformat"]["repetitions"] is 1:        del cleanconfig["group_testset_reformat"]["repetitions"]
        if not that.config["group_testset_reformat"]["dynamic_repetitions"]: del cleanconfig["group_testset_reformat"]["dynamic_repetitions"]
        if not that.config["group_testset_reformat"]:                        del cleanconfig["group_testset_reformat"]
        return cleanconfig

class ActionTrainModel(DefaultActionObject):
    """docstring for ActionTrainModel"""
    def __init__(self, that):
        self.that = that

    def initialise_config(self):
        that=self.that
        raise NotImplementedError

    def process_args(self, args):
        that=self.that
        raise NotImplementedError

    def valid(self):
        that=self.that
        throw_error = False
        if not that.config["group_exec"]["oxcg_bin"]: throw_error = True

        raise NotImplementedError
        return not throw_error

    def config_clean_copy(self, cleanconfig):
        that=self.that
        raise NotImplementedError

class ActionEvaluateModel(DefaultActionObject):
    """docstring for ActionTrainModel"""
    def __init__(self, that):
        self.that = that

    def initialise_config(self):
        that=self.that
        raise NotImplementedError

    def process_args(self, args):
        that=self.that
        raise NotImplementedError

    def valid(self):
        that=self.that
        throw_error = False
        if not that.config["group_exec"]["oxcg_bin"]: throw_error = True

        raise NotImplementedError
        return not throw_error

    def config_clean_copy(self, cleanconfig):
        that=self.that
        raise NotImplementedError

def get_estimates(theta_source, joint_target):

    # Get evaluation lines
    source_lines = [line.strip() for line in theta_source]
    target_lines = [line.strip() for line in joint_target]

    assert len(source_lines) == len(target_lines), "Source and target not well aligned"

    zlines = zip(source_lines,target_lines)
    zlines==zlines

    # [(theta_k, sentence_k)]

def run_evaluation():
    pass

def main(argv):
    DefaultActionObject(argv)

if __name__ == '__main__':
    main(sys.argv[1:])