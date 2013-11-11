import sys
import argparse
from itertools import chain, izip
from os.path import join as pjoin

REVISION = "$Rev: 2 $"

class DefaultHelpParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def appendTag(l, tag):
    parts = l.split()
    parts = map(lambda x: x + '_' + tag, parts)
    return " ".join(parts)


def main():
    parser = DefaultHelpParser(description='Reformat a source and target corpus for use in a joint-probability CNLM. In addition, this script generates files for testing paraphrases.', epilog="Copyright 2013 Ed Grefenstette and Karl Moritz Hermann, %s" % REVISION)

    parser.add_argument('source', metavar='SOURCE_CORPUS', type=str,
            help='Original source corpus (labels/source sentences).')
    parser.add_argument('target', metavar='TARGET_CORPUS', type=str,
            help='Original target corpus (target sentences).')
    parser.add_argument('-o', '--output', dest='outputdir', default='.',
            help='Output directory for new source/target files.')
    parser.add_argument('-p', '--prefix-eval-files', dest='prefix',
            default='test',
            help='Prefix for source and target evaluation files.')
    parser.add_argument('--source-eos-label', dest='eos_source', type=str, default=None,
            help='Optional end of sentence label for source sentence.')
    parser.add_argument('--target-eos-label', dest='eos_target', type=str, default=None,
            help='Optional end of sentence label for target sentence.')

    args = parser.parse_args()

    source_lines = [appendTag(line.strip(), 'source') for line  in
            open(args.source, 'r') if len(line.strip()) > 0 and not
            line.strip().startswith('#')]
    target_lines = [appendTag(line.strip(), 'target') for line  in
            open(args.target, 'r') if len(line.strip()) > 0 and not
            line.strip().startswith('#')]
    assert len(source_lines) == len(target_lines)

    ## Add source EOS symbol if specified
    if args.eos_source:
        source_lines = [line + " " + args.eos_source for line in source_lines]

    ## Add target EOS symbol if specified
    if args.eos_target:
        target_lines = [line + " " + args.eos_target for line in target_lines]

    enumeration = range(len(source_lines))
    enumeration_source = ["us%d" % num for num in enumeration]
    enumeration_target = ["ut%d" % num for num in enumeration]
    new_source_labels = list(chain.from_iterable(izip(enumeration_source, enumeration_target)))
    new_target_labels = list(chain.from_iterable(izip(source_lines, target_lines)))

    with open(pjoin(args.outputdir, "%s_testtrain_source" % args.prefix), 'w') as f:
        for line in new_source_labels:
            f.write("%s\n" % line)

    with open(pjoin(args.outputdir, "%s_testtrain_target" % args.prefix), 'w') as f:
        for line in new_target_labels:
            f.write("%s\n" % line)

    # Magic: build evaluation files for source and target
    for group in [("source", source_lines, enumeration_target), ("target", target_lines, enumeration_source)]:
        name = "%s_%s" % (args.prefix, group[0])
        data_lines = group[1]
        reference_lines = group[2]
        with open(pjoin(args.outputdir, "%s_data" % name), 'w') as f:
            for line in data_lines:
                f.write("%s\n" % line)
        with open(pjoin(args.outputdir, "%s_reference" % name), 'w') as f:
            for line in reference_lines:
                f.write("%s\n" % line)
        with open(pjoin(args.outputdir, "%s_candidates" % name), 'w') as f:
            for line in range(len(reference_lines)):
                for offset in range(-5, 5):
                    f.write("%s " % reference_lines[ (line + offset) %
                        len(reference_lines)])
                f.write("\n")


if __name__ == '__main__':
    main()
