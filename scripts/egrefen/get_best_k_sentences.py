from __future__ import division
from operator import itemgetter
import sys

import numpy as np
from math import sqrt

def get_k_best_sids(sid2vector, k=5, cosine=False):
    sid2bestsids = {}
    for sid in sid2vector:
        vector = sid2vector[sid]

        score_list = []

        for other_sid in sid2vector:

            if sid == other_sid:
                continue

            other_vector = sid2vector[other_sid]
            score = np.dot(vector, other_vector)

            if cosine:
                vector_norm = sqrt(np.dot(vector, vector))
                other_norm = sqrt(np.dot(other_vector,other_vector))
                score = score/(vector_norm * other_norm)

            score_list.append((other_sid, score))

        best_k = sorted(score_list, key=itemgetter(1), reverse=True)[:k]
        sid2bestsids[sid] = map(itemgetter(0), best_k)
    return sid2bestsids

def print_best_sentences(sid2sent, sid2bestsids, ostream=sys.stdout):
    for sid in sid2sent:
        ostream.write("REF: %s\n" % sid2sent[sid])
        for i, other_sid in enumerate(sid2bestsids[sid]):
            ostream.write("%d: %s\n" % (i, sid2sent[other_sid]))
        ostream.write("---------\n")


