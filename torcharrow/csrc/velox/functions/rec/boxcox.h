#pragma once

#include <cmath>
#include <velox/functions/Macros.h>
#include "velox/core/QueryConfig.h"
#include "velox/expression/ComplexViewTypes.h"
#include "velox/functions/Udf.h"
#include "velox/type/Type.h"


namespace facebook::torcharrow::functions {

    template <typename TExecParams>
    struct boxcox {
        VELOX_DEFINE_FUNCTION_TYPES(TExecParams);

        // By default we use precision = 63, to computed signed division quotient
        // rouded towards negative infinity.

        template <typename TOutput, typename TInput>
        FOLLY_ALWAYS_INLINE void callNullFree(
            TOutput& result,
            const TInput& values,
            const float& lambda) 
        {
            if constexpr (std::is_same<TInput, float>::value) {
                if (values > 0){
                    if(lambda == 0.0f)
                        result = log(values);
                    else
                        result = (pow(values, lambda) - 1.0f) / lambda;
                }
                else{
                    result = values;
                }
            }
            else{ // list of values
                for (const auto& val : values) {
                    if(val > 0){
                        if(lambda == 0.0f)
                            result.push_back(log(val));
                        else
                            result.push_back((pow(val, lambda) - 1.0f) / lambda);
                    }
                    else{
                        result.push_back(val);
                    }
                }
            }
        }
    };

} // namespace facebook::torcharrow::functions