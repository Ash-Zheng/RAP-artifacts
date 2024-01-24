/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <velox/functions/Macros.h>
#include "velox/core/QueryConfig.h"
#include "velox/expression/ComplexViewTypes.h"
#include "velox/functions/Udf.h"
#include "velox/type/Type.h"

namespace facebook::torcharrow::functions {

template <typename T>
struct ngram {
  VELOX_DEFINE_FUNCTION_TYPES(T);

    template <typename TOutput, typename TInput>
    FOLLY_ALWAYS_INLINE void callNullFree(
      TOutput& result,
      const TInput& values,
      const int32_t& n_of_gram) {

        VELOX_CHECK(n_of_gram > 0, "n_of_gram should not be zero.");
        int32_t actual_n_of_gram = std::min(n_of_gram, values.size());
        int32_t final_length = (values.size() - actual_n_of_gram + 1) * actual_n_of_gram;
        result.reserve(final_length);

        for (int i = 0; i < values.size() - actual_n_of_gram + 1; i++){
            for (int j = 0; j < actual_n_of_gram; j++){
                result.push_back(values[i + j]);
            }
        }
    }
};

} // namespace facebook::torcharrow::functions
