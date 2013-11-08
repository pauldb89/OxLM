// Copyright 2013 Karl Moritz Hermann
// File: paraphrase.cc
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 08-11-2013
// Last Update: Fri 08 Nov 2013 11:41:09 AM GMT

#include "experimental/kmh/paraphrase.h"
#include "cg/cnlm.h"

using namespace std;

namespace oxlm {

// namespace experiments {

void paraphrase(CNLMBase::WordVectorsType& source,
                CNLMBase::WordVectorsType& target, bool cosine) {

  // Compare all sentence level representations and pick most similar.
  int correct_count = 0;
  int closest = 0;
  double minimum = 0;
  double distance = 0;

  // Need to get length!
  int length = source.rows();
  for (auto a = 0; a < length; ++a) {
    int start = (a - 5 + length) % length;
    if (cosine) {
      // Cosine distance (complement to cosine similarity).
      minimum = 1.0 - ((source.row(a).transpose() * target.row(start)).sum() /
                       (source.row(a).norm() * target.row(start).norm()));
    } else {
      // Euclidean distance.
      minimum = (source.row(a) - target.row(start)).squaredNorm();
    }
    closest = start;
    for (auto c = 1; c < 10; ++c) {
      auto b = (start + c) % length;
      if (cosine) {
        distance = 1.0 - ((source.row(a).transpose() * target.row(b)).sum() /
                          (source.row(a).norm() * target.row(b).norm()));
      } else {
        distance = (source.row(a) - target.row(b)).squaredNorm();
      }

      // If we have a new minimum, update this accordingly.
      if (distance < minimum) {
        closest = b;
        minimum = distance;
      }
    }
    // If the index of the closest element matches a, the paraphrase was
    // identified correctly.
    if ( a == closest)
      ++correct_count;
  }

  cout << "PP10 Correct: " << correct_count << "/"
    << source.rows() << " " << (100.0 * correct_count)/source.rows()
    << "%" << endl;
}

// }  // namespace experiments
}  // namespace oxlm
