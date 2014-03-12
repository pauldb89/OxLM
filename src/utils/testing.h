#pragma once

#include "gtest/gtest.h"

namespace oxlm {

#define EXPECT_MATRIX_NEAR(m1, m2, abs_error) \
    EXPECT_LE((m1 - m2).cwiseAbs().maxCoeff(), abs_error)

} // namespace oxlm
