import sys
import argparse
from itertools import chain, izip
from os.path import join as pjoin

REVISION = "$Rev: 1 $"

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

    parser.add_argument('source', metavar='SOURCE_CORPUS', type=str,
                        help='Original source corpus (labels/source sentences).')
    parser.add_argument('target', metavar='TARGET_CORPUS', type=str,
                        help='Original target corpus (target sentences).')
    parser.add_argument('-o', '--output', dest='outputdir', default='.',
                        help='Output directory for new source/target files.')
    parser.add_argument('-s', '--output-source-file', dest='outputsource', default='source_entries',
                        help='Output filename for new source file.')
    parser.add_argument('-t', '--output-target-file', dest='outputtarget', default='target_entries',
                        help='Output filename for new target file.')


    args = parser.parse_args()

    source_lines = [appendTag(line.strip(), 'source') for line  in open(args.source, 'r') if len(line.strip()) > 0 and not line.strip().startswith('#')]
    target_lines = [appendTag(line.strip(), 'target') for line  in open(args.target, 'r') if len(line.strip()) > 0 and not line.strip().startswith('#')]
    assert len(source_lines) == len(target_lines)

    enumeration = range(len(source_lines))
    numbers = list(chain.from_iterable(izip(enumeration, enumeration)))
    new_source_labels = ["s%d" % num for num in numbers]
    new_target_labels = list(chain.from_iterable(izip(source_lines, target_lines)))

    with open(pjoin(args.outputdir, args.outputsource), 'w') as f:
        for line in new_source_labels:
            f.write("%s\n" % line)

    with open(pjoin(args.outputdir, args.outputtarget), 'w') as f:
        for line in new_target_labels:
            f.write("%s\n" % line)

if __name__ == '__main__':
    main()