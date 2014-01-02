from __future__ import division
import argparse
from operator import itemgetter
from numpy import array, dot
from numpy.linalg import norm

def cosine(u,v):
    return dot(u,v)/(norm(u)*norm(v))

def get_top_k(index, classes, vectors, k):
    scores = zip(classes, map(lambda v: cosine(vectors[index], v), vectors))
    del scores[index]
    scores.sort(key=itemgetter(1))
    scores.reverse()
    return [x[0] for x in scores][:k]

def get_prec(classes, vectors, k):
    return sum([1 for x in range(len(classes)) if classes[x] in get_top_k(x, classes, vectors, k)])/len(classes)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vectorfile", type=argparse.FileType())
    parser.add_argument("sid2class", type=argparse.FileType())
    parser.add_argument("-k", dest="k", type=int, default=5)

    args = parser.parse_args()

    sid2class = dict([line.strip().split() for line in args.sid2class])

    lines = [(sid2class[x[0]], array(map(float,x[1:]))) for x in [line.strip().split() for line in args.vectorfile]]

    classes = [sid for (sid, vect) in lines]
    vectors = array([vect for (sid, vect) in lines])

    print "Top-5 precision:", 100*get_prec(classes, vectors, args.k)

if __name__ == '__main__':
    main()
