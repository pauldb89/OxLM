import sys
import argparse
# import subprocess
# import shlex

from copy import deepcopy as copy

REVISION = "$Rev: 1 $"

class DefaultHelpParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n\n' % message)
        self.print_help()
        sys.exit(2)

def process_args(args):

    ## Build arg config
    config = {}

    config["group_exec"] = {}
    config["group_exec"]["python_script_path"] = None                   # Mandatory
    config["group_exec"]["oxcg_bin"] = None                             # Mandatory

    config["group_testset_reformat"] = {}
    config["group_testset_reformat"]["test_sentences"] = None           # Mandatory
    config["group_testset_reformat"]["output_source"] = None            # Mandatory
    config["group_testset_reformat"]["output_target"] = None            # Mandatory
    config["group_testset_reformat"]["labels"] = None                   # Mandatory
    config["group_testset_reformat"]["repetitions"] = 1                 # Optional
    config["group_testset_reformat"]["dynamic_repetitions"] = False     # Optional


    ## Load configuration if specified

    if args.config:
        from yaml import load as yload
        try:
            from yaml import CLoader as YLoader
        except ImportError:
            from yaml import Loader as YLoader
        existing_config = yload(open(args.config), YLoader)

        for groupkey in existing_config:
            for key in existing_config[groupkey]:
                config[groupkey][key] = existing_config["group_exec"][key]


    ## Override config with command line args, if non-null

    if args.python_script_path:
        config["group_exec"]["python_script_path"] = args.python_script_path
    if args.oxcg_bin:
        config["group_exec"]["oxcg_bin"] = args.oxcg_bin

    if args.test_sentences:
        config["group_testset_reformat"]["test_sentences"] = args.test_sentences
    if args.output_source:
        config["group_testset_reformat"]["output_source"] = args.output_source
    if args.output_target:
        config["group_testset_reformat"]["output_target"] = args.output_target
    if args.labels:
        config["group_testset_reformat"]["labels"] = args.labels
    if args.repetitions:
        config["group_testset_reformat"]["repetitions"] = args.repetitions
    if args.dynamic_repetitions:
        config["group_testset_reformat"]["dynamic_repetitions"] = args.dynamic_repetitions

    return config

def valid(config, command):
    ## Validate args
    throw_error = command in ['t','e','p','tp', 'pe', 'te', 'tpe']

    if not config["group_exec"]["python_script_path"]: throw_error = True
    if not config["group_exec"]["oxcg_bin"]: throw_error = True

    if 'p' in command:
        if not config["group_testset_reformat"]["test_sentences"]: throw_error = True
        if not config["group_testset_reformat"]["output_source"]: throw_error = True
        if not config["group_testset_reformat"]["output_target"]: throw_error = True
        if not config["group_testset_reformat"]["labels"]: throw_error = True

    return not throw_error

def saveconfig(config, configfile):
    config = copy(config)

    # Clean group_exec
    if not config["group_exec"]["python_script_path"]:              del config["group_exec"]["python_script_path"]
    if not config["group_exec"]["oxcg_bin"]:                        del config["group_exec"]["oxcg_bin"]
    if not config["group_exec"]:                                    del config["group_exec"]

    if not config["group_testset_reformat"]["test_sentences"]:      del config["group_testset_reformat"]["test_sentences"]
    if not config["group_testset_reformat"]["output_source"]:       del config["group_testset_reformat"]["output_source"]
    if not config["group_testset_reformat"]["output_target"]:       del config["group_testset_reformat"]["output_target"]
    if not config["group_testset_reformat"]["labels"]:              del config["group_testset_reformat"]["labels"]
    if config["group_testset_reformat"]["repetitions"] is 1:        del config["group_testset_reformat"]["repetitions"]
    if not config["group_testset_reformat"]["dynamic_repetitions"]: del config["group_testset_reformat"]["dynamic_repetitions"]
    if not config["group_testset_reformat"]:                        del config["group_testset_reformat"]

    from yaml import dump as ydump
    try:
        from yaml import CDumper as YDumper
    except ImportError:
        from yaml import Dumper as YDumper
    ydump(config, configfile, YDumper, default_flow_style=False)

def print_config(config):

    if config["group_testset_reformat"]["labels"]:
        labels = ", ".join(config["group_testset_reformat"]["labels"])
    else:
        labels = None

    output="""
    #############################################################
    #############################################################
    ## Path options
    ## ------------
    ## python script path = %s
    ## oxlm bin path = %s
    #############################################################
    ## Test set preprocessing options
    ## ------------------------------
    ## test sentences = %s
    ## output source = %s
    ## input source = %s
    ## labels = %s
    ## repetitions = %d
    ## dynamic repretitions = %s
    #############################################################
    #############################################################
    """ % (
            ## path options
            config["group_exec"]["python_script_path"],
            config["group_exec"]["oxcg_bin"],

            ## preprocessing options
            config["group_testset_reformat"]["test_sentences"],
            config["group_testset_reformat"]["output_source"],
            config["group_testset_reformat"]["output_target"],
            labels,
            config["group_testset_reformat"]["repetitions"],
            config["group_testset_reformat"]["dynamic_repetitions"]
        )

    sys.stderr.write(output)

def process_testset():
    pass

def process_dataset():
    pass

def train_model():
    pass

def run_evaluation():
    pass

def main():

    parser = DefaultHelpParser(description='Run CNLM label experiments.', epilog="Copyright 2013 Ed Grefenstette, %s" % REVISION)
    parser.add_argument('-c', '--config', help="Load configuration file CONFIG.", default=None)
    parser.add_argument('--save-config', dest="save_config_path", metavar="CONFIGPATH", help="Save configuration and quit.", type=str, default=None)
    parser.add_argument('--save-config-and-continue', dest="saveandcontinue_config_path", metavar="CONFIGPATH", help="Save configuration and quit.", type=str, default=None)
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help="Verbose output to stderr.")

    group_exec = parser.add_argument_group("executable/script paths")

    group_exec.add_argument('--python-scripts', dest="python_script_path", help="Path for python helper scripts.", default=None)
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

    args = parser.parse_args()

    config = process_args(args)

    if args.verbose:
        print_config(config)

    if args.save_config_path or args.saveandcontinue_config_path:
        if args.save_config_path:
            configfile = open(args.save_config_path, 'w')
        else:
            configfile = open(args.saveandcontinue_config_path, 'w')
        saveconfig(config, configfile)
        configfile.close()
        if args.saveandcontinue_config_path:
            if args.verbose:
                sys.stderr.write("Saved configuration to %s and continuing.\n" % args.save_config_path)
        else:
            if args.verbose:
                sys.stderr.write("Saved configuration to %s and quitting.\n" % args.save_config_path)
            sys.exit(0)

    if not (valid(config, args.command)):
        parser.error("Not enough arguments.")

    must_train = 't' in args.command
    must_preprocess = 'p' in args.command
    must_evaluate = 'e' in args.command

    if must_train:
        raise NotImplementedError("Training not implemented yet")
    if must_preprocess:
        raise NotImplementedError("")
    if must_evaluate:
        raise NotImplementedError

if __name__ == '__main__':
    main()