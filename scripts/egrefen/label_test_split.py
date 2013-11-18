import sys
import argparse

REVISION = "$Rev: 2 $"

class DefaultHelpParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n\n' % message)
        self.print_help()
        sys.exit(2)


def label_train_split(labels, label_repetitions, istream, ostream_source, ostream_target, dynamic_repetitions):

    num_labels = len(labels)

    test_sentences = [line.strip() for line in istream if len(line.strip()) > 0 and not line.strip().startswith('#')]
    num_test_sentences = len(test_sentences)

    for k in range(num_test_sentences):
        for i in range(num_labels):
            ostream_source.write('s(%d,%d)\n' % (k, i))
            ostream_target.write('%s\n' % test_sentences[k])
            if dynamic_repetitions:
                label_repetitions = len(test_sentences[k].split())
            for n in range(label_repetitions):
                ostream_source.write('s(%d,%d)\n' % (k, i))
                ostream_target.write('%s\n' % labels[i])


def main():

    parser = DefaultHelpParser(description='Reformat a test sentence file for use with a joint-probability CNLM.', epilog="Copyright 2013 Ed Grefenstette, %s" % REVISION)

    parser.add_argument('sentences', type=str,
                        help='Test sentences (one per line).')
    parser.add_argument('source', type=str,
                        help='Output source file.')
    parser.add_argument('target', type=str,
                        help='Output target file.')
    parser.add_argument('-l', '--labels', dest='labels', default=None,
                        help='List of comma-separated labels, of the form "label1,label2,...,labeln".')
    parser.add_argument('-r', '--repetitions', dest='repetitions', default=1, type=int,
                        help='Number of times the label lines are repeated (advanced feature).')
    parser.add_argument('-d', '--dynamic-repetitions', dest='dynamic_repetitions', default=False, action='store_true',
                        help='Base number of times label lines are repeated on the length of the source sentence (advanced feature).')

    args = parser.parse_args()

    if not args.labels:
        sys.stderr.write('error: A list of labels must be provided by the -l flag, of the form "label1,label2,...,labeln".\n\n')
        parser.print_help()
        sys.exit(2)

    ## Read CLI Parameters
    test_sentence_path = args.sentences
    source_path = args.source
    target_path = args.target
    label_repetitions = args.repetitions
    dynamic_repetitions = args.dynamic_repetitions
    labels = args.labels.split(',')

    istream = open(test_sentence_path)
    ostream_source = open(source_path, 'w')
    ostream_target = open(target_path, 'w')

    ## Split training set
    label_train_split(labels, label_repetitions, istream, ostream_source, ostream_target, dynamic_repetitions)

    ## Cleanup
    istream.close()
    ostream_source.close()
    ostream_target.close()

if __name__ == '__main__':
    main()