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
#include <vector>

namespace facebook::torcharrow::functions {

template <typename T>
struct mapid {
  VELOX_DEFINE_FUNCTION_TYPES(T);

    template <typename TOutput, typename TInput, typename TMap>
    FOLLY_ALWAYS_INLINE void callNullFree(
      TOutput& result,
      const TInput& values,
      const TMap& _map
      ){

        int32_t length = _map.size();
        if constexpr (std::is_same<TInput, float>::value) {
            if (values >= length)
                result = 0;
            else
                result = _map[values];
        }
        else{
            for (const auto& val : values) {
                if (val >= length)
                    result.push_back(0);
                else
                    result.push_back(_map[val]);
            }
        }
    }
};

} // namespace facebook::torcharrow::functions
