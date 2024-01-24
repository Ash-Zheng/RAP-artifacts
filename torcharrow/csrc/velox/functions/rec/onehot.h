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

inline int32_t computeOneHotIdx(
      const float& lower,
      const float& upper,
      const int32_t& num_class,
      const float& val
      ) {

    float step = (upper - lower) / float(num_class);
    int32_t index = int((val - lower) / step);
    if (index < 0 || index >= num_class) {
      return 0;
    } 
    else
        return index;
}

template <typename T>
struct onehot {
  VELOX_DEFINE_FUNCTION_TYPES(T);

    template <typename TOutput, typename TInput>
    FOLLY_ALWAYS_INLINE void callNullFree(
      TOutput& result,
      const TInput& values,
      const float& lower,
      const float& upper,
      const int32_t& num_class) {

        VELOX_CHECK(num_class > 0, "number of classses should not be zero.");
        VELOX_CHECK(upper > lower, "upper bound should larger than lower bound.");
        auto idx = computeOneHotIdx(lower, upper, num_class, values[0]);
        result.reserve(int(num_class));
        for (int i = 0; i < int(num_class); i++){
            if (i == idx)
                result.push_back(1.0);
            else
                result.push_back(0.0);
        }
    }
};

} // namespace facebook::torcharrow::functions
