#pragma once

#include <cmath>
#include <velox/functions/Macros.h>
#include "velox/core/QueryConfig.h"
#include "velox/expression/ComplexViewTypes.h"
#include "velox/functions/Udf.h"
#include "velox/type/Type.h"


namespace facebook::torcharrow::functions {

    template <typename TExecParams>
    struct logit {
        VELOX_DEFINE_FUNCTION_TYPES(TExecParams);

        // By default we use precision = 63, to computed signed division quotient
        // rouded towards negative infinity.

        template <typename TOutput, typename TInput>
        FOLLY_ALWAYS_INLINE void callNullFree(
            TOutput& result,
            const TInput& values,
            const float& eps) 
        {
            if constexpr (std::is_same<TInput, float>::value) {
            // constexpr (std::is_same<TInput, float>::values) {
                float tmp = values;
                if(values < eps)
                    tmp = eps;
                else if(tmp > 1.0f-eps)
                    tmp = 1.0f-eps;
                result = log(tmp/(1.0f-tmp));
            }
            else{ // list of values
                for (const auto& val : values) {
                    float tmp = val;
                    if(val < eps)
                        tmp = eps;
                    else if(tmp > 1.0f-eps)
                        tmp = 1.0f-eps;

                    result.push_back(log(tmp/(1.0f-tmp)));
                }
            }
        }
    };

} // namespace facebook::torcharrow::functions