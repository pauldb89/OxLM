// Copyright 2013 Karl Moritz Hermann
// File: paraphrase.h
// Author: Karl Moritz Hermann (mail@karlmoritz.com)
// Created: 08-11-2013
// Last Update: Fri 08 Nov 2013 11:39:27 AM GMT

#ifndef EXPERIMENTAL_KMH_PARAPHRASE_H
#define EXPERIMENTAL_KMH_PARAPHRASE_H

#include "cg/cnlm.h"

namespace oxlm {

// namespace experiments {

void paraphrase(CNLMBase::WordVectorsType& source,
                CNLMBase::WordVectorsType& target, bool cosine);

// }  // namespace experiments
}  // namespace oxlm

#endif  // EXPERIMENTAL_KMH_PARAPHRASE_H
