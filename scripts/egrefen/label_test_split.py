import sys
import argparse

REVISION = "$Rev: 1 $"

class DefaultHelpParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n\n' % message)
        self.print_help()
        sys.exit(2)

def main():

    parser = DefaultHelpParser(description='Reformat a test sentence file for use with a joint-probability CNLM.', epilog="Copyright 2013 Ed Grefenstette, %s" % REVISION)

    parser.add_argument('sentences', type=str,
                        help='Test sentences (one per line).')
    parser.add_argument('-l', '--labels', dest='labels', default=None,
                        help='List of comma-separated labels, of the form "label1,label2,...,labeln".')
    parser.add_argument('-r', '--repetitions', dest='repetitions', default=1, type=int,
                        help='Number of times the label lines are repeated (advanced feature).')
    parser.add_argument('-o','--output', type=str, default=None,
                        help='Output path.')


    args = parser.parse_args()

    if not args.labels:
        sys.stderr.write('error: A list of labels must be provided by the -l flag, of the form "label1,label2,...,labeln".\n\n')
        parser.print_help()
        sys.exit(2)

    test_sentence_path = args.sentences
    output_path = args.output

    label_repetitions = args.repetitions

    labels = args.labels.split(',')
    num_labels = len(labels)

    test_sentences = [line.strip() for line in open(test_sentence_path) if len(line.strip()) > 0 and not line.strip().startswith('#')]

    num_test_sentences = len(test_sentences)

    if output_path:
        output = open(output_path, 'w')
    else:
        output = sys.stdout

    for k in range(num_test_sentences):
        for i in range(num_labels):
            output.write('s(%d,%d) %s\n' % (k, i, test_sentences[k]))
            for n in range(label_repetitions):
                output.write('s(%d,%d) %s\n' % (k, i, labels[i]))

    if output_path:
        output.close()

if __name__ == '__main__':
    main()