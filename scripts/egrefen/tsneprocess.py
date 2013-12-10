#!/usr/bin/env python

import sys
import argparse
from os.path import join as pjoin
from os.path import abspath

from numpy import array

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--perplexity', type=float, default=30.0,
                        help='Perplexity for t-sne.')
    parser.add_argument('-l','--perplexity-list', dest="plist", type=str, default=None,
                        help='Comma separated perplexity list.')
    parser.add_argument('-d','--dimensions', type=int, default=None,
                        help='Dimensions of input vectors.')
    parser.add_argument('-t','--vectors', type=str, default=None,
                        help='Path to input vector file.')
    parser.add_argument('--path', type=str, default=None,
                        help='Path to t-sne scripts.')
    parser.add_argument('-o', '--output', type=str, default='.',
                        help='Output directory.')

    args = parser.parse_args()

    if args.path is None or args.vectors is None or args.dimensions is None:
        parser.print_help()
        sys.exit(1)

    if args.plist is None:
        perplexities = [args.perplexity]
    else:
        perplexities = [float(p) for p in args.plist.split(',')]

    sys.path.append(abspath(args.path))
    import calc_tsne as tsne

    TRAINING_THETAS = args.vectors
    dimensions = args.dimensions

    lines = [line.strip().split() for line in open(TRAINING_THETAS)]

    labels = [line[0] for line in lines]
    vectors = array([map(float, line[1:]) for line in lines])

    for perplexity in perplexities:
        X = tsne.tsne(vectors, initial_dims = dimensions, perplexity = perplexity)
        ofilename = "Xpp%f.vectors" % perplexity
        with open(pjoin(args.output, ofilename), 'w') as f:
            for i in range(len(labels)):
                f.write("%s %s\n" % (labels[i], " ".join(map(str,X[i,:]))))

if __name__ == '__main__':
    main()
