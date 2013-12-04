import sys
import argparse
from itertools import chain, izip
from os.path import join as pjoin

REVISION = "$Rev: 4 $"

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
    parser = DefaultHelpParser(description='Reformat a source and target corpus for use in a joint-probability CNLM.', epilog="Copyright 2013 Ed Grefenstette, %s" % REVISION)

    parser.add_argument('-s','--source', metavar='SOURCE_CORPUS', type=str, default=None,
                        help='Original source corpus (labels/source sentences).')
    parser.add_argument('-t','--target', metavar='TARGET_CORPUS', type=str,
                        help='Original target corpus (target sentences).')
    parser.add_argument('-o', '--output', dest='outputdir', default='.',
                        help='Output directory for new source/target files.')
    parser.add_argument('--output-source-file', dest='outputsource', default='source_entries',
                        help='Output filename for new source file.')
    parser.add_argument('--output-target-file', dest='outputtarget', default='target_entries',
                        help='Output filename for new target file.')
    parser.add_argument('--source-eos-label', dest='eos_source', type=str, default=None,
            help='Optional end of sentence label for source sentence.')
    parser.add_argument('--target-eos-label', dest='eos_target', type=str, default=None,
            help='Optional end of sentence label for target sentence.')
    parser.add_argument('--no-target-subscripts', dest='no_target_subscripts', action="store_true", default=False,
            help='Do not add subscripts to target.')


    args = parser.parse_args()

    if args.source:
        source_lines = [appendTag(line.strip(), 'source') for line  in open(args.source, 'r') if len(line.strip()) > 0]

    if args.no_target_subscripts and not args.source:
        target_lines = [line.strip() for line  in open(args.target, 'r') if len(line.strip()) > 0]
    else:
        target_lines = [appendTag(line.strip(), 'target') for line  in open(args.target, 'r') if len(line.strip()) > 0]

    if args.source:
        assert len(source_lines) == len(target_lines)

    ## Add source EOS symbol if specified
    if args.source and args.eos_source:
        source_lines = [line + " " + args.eos_source for line in source_lines]

    ## Add target EOS symbol if specified
    if args.eos_target:
        target_lines = [line + " " + args.eos_target for line in target_lines]

    enumeration = range(len(target_lines))

    if args.source:
        numbers = list(chain.from_iterable(izip(enumeration, enumeration)))
    else:
        numbers = enumeration

    new_source_labels = ["s%d" % num for num in numbers]

    if args.source:
        new_target_labels = list(chain.from_iterable(izip(source_lines, target_lines)))
    else:
        new_target_labels = target_lines

    with open(pjoin(args.outputdir, args.outputsource), 'w') as f:
        for line in new_source_labels:
            f.write("%s\n" % line)

    with open(pjoin(args.outputdir, args.outputtarget), 'w') as f:
        for line in new_target_labels:
            f.write("%s\n" % line)

if __name__ == '__main__':
    main()
