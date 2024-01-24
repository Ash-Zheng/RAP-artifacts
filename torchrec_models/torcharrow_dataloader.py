#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torcharrow as ta
import torcharrow.dtypes as dt
import torcharrow.pytorch as tap
from torch.utils.data import DataLoader
from torcharrow import functional
from torchdata.datapipes.iter import FileLister
from torchrec.datasets.criteo import (
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    DEFAULT_LABEL_NAME,
)

from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

class _DenseTensorConversion(tap.TensorConversion):
    def to_tensor(self, df: ta.DataFrame):
        # concate all columns into a single tensor
        keys = df.columns
        tensor_list = []
        for name in keys:
            column = df[name]
            tansor_column = torch.tensor(column)
            if len(tansor_column.shape) == 1:
                tansor_column = tansor_column.unsqueeze(1)
            tensor_list.append(tansor_column)
        
        tensor = torch.cat(tensor_list, dim=1)
        return tensor

class _JaggedTensorConversion(tap.TensorConversion):
    # pyre-fixme[14]: `to_tensor` overrides method defined in `TensorConversion`
    #  inconsistently.
    def to_tensor(self, df: ta.DataFrame):
        kjt_keys = df.columns
        kjt_values = []
        kjt_lengths = []
        for name in kjt_keys:
            column = df[name]
            for value in column:
                kjt_values.extend(value)
                kjt_lengths.append(len(value))

        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=kjt_keys,
            values=torch.tensor(kjt_values),
            lengths=torch.tensor(kjt_lengths),
        )
        return kjt

class _JaggedTensorConversionSkip(tap.TensorConversion):
    # pyre-fixme[14]: `to_tensor` overrides method defined in `TensorConversion`
    #  inconsistently.
    def to_tensor(self, df: ta.DataFrame):
        return None


class _Scalar(tap.TensorConversion):
    def to_tensor(self, df: ta.DataFrame):
        labels = torch.tensor(df)
        return labels


class torcharrow_dataloader():
    def __init__ (self, parquet_directory, args, device, preprocessing_plan=0):
        self.args = args
        self.device = device

        # TODO support batch_size for load_parquet_as_df.
        # TODO use OSSArrowDataPipe once it is ready
        self.source_dp = FileLister(parquet_directory, masks="*.parquet")
        self.parquet_df_dp = self.source_dp.load_parquet_as_df()

        self.map = ta.column([list(range(16,1,-1)) for _ in range(args.batch_size)])
        self.preprocessing_plan = preprocessing_plan

        # self.parquet_df_dp = source_dp.load_parquet_as_df()
        # self.parquet_list = list(self.parquet_df_dp)

    
    def preproc(self, df, salt=0):
        for feature_name in DEFAULT_INT_NAMES:
            df[feature_name] = df[feature_name].fill_null(0)
            df[feature_name] = df[feature_name].cast(dt.float32)
        for feature_name in DEFAULT_CAT_NAMES:
            df[feature_name] = df[feature_name].fill_null(0)
            df[feature_name] = df[feature_name].cast(dt.int64)

        # construct a sprase index from a dense one
        df["bucketize_int_0"] = functional.bucketize(df["int_0"], [0.5, 1.0, 1.5]).cast(
            dt.int64
        )

        for idx, int_name in enumerate(self.args.int_name):
            df[int_name] = functional.logit(df[int_name], 1e-5)

        # flatten several columns into one
        df["dense_features"] = ta.dataframe(
            {int_name: df[int_name] for int_name in DEFAULT_INT_NAMES}
        )

        for idx, cat_name in enumerate(self.args.cat_name):
            # hash our embedding index into our embedding tables
            df[cat_name] = functional.sigrid_hash(df[cat_name], salt, self.args.num_embeddings_per_feature[idx])
            df[cat_name] = functional.array_constructor(df[cat_name])
            df[cat_name] = functional.firstx(df[cat_name], 1)

        df["sparse_features"] = ta.dataframe(
            {
                cat_name: df[cat_name]
                for cat_name in self.args.cat_name
            }
        )

        df = df[["dense_features", "sparse_features", DEFAULT_LABEL_NAME]]

        return df

    
    def preproc_1(self, df):
        for feature_name in DEFAULT_INT_NAMES:
            df[feature_name] = df[feature_name].fill_null(0)
            df[feature_name] = df[feature_name].cast(dt.float32)
        for feature_name in DEFAULT_CAT_NAMES:
            df[feature_name] = df[feature_name].fill_null(0)
            df[feature_name] = df[feature_name].cast(dt.int64)

        # ========================================================
        for i in range(self.args.nDense):
            df["int_{}".format(i)] = df["int_{}".format(i%13)]
        for i in range(self.args.nSparse):
            df["cat_{}".format(i)] = df["cat_{}".format(i%26)]

        logit_list = ["int_{}".format(i) for i in range(0,7)]
        boxcox_list = ["int_{}".format(i) for i in range(7,16)]
        onehot_list = ["int_{}".format(i) for i in range(16,26)]

        for i in range(4):
            for int_name in logit_list:
                df[int_name] = functional.logit(df[int_name], 1e-5)
            for int_name in boxcox_list:
                df[int_name] = functional.boxcox(df[int_name], 1e-5)
            for int_name in onehot_list:
                df[int_name + "_onehot"] = functional.array_constructor(df[int_name])
                df[int_name + "_onehot"] = functional.onehot(df[int_name + "_onehot"], 0.0, 16.0, 5)
                df[int_name + "_onehot"] = functional.firstx(df[int_name + "_onehot"], 1)

            sigrid_hash_list = ["cat_{}".format(i) for i in range(0,11)]
            clamp_list = ["cat_{}".format(i) for i in range(11,27)]
            firstx_list = ["cat_{}".format(i) for i in range(27,39)]
            ngram_list = ["cat_{}".format(i) for i in range(39,52)]

            for cat_name in sigrid_hash_list:
                df[cat_name] = functional.sigrid_hash(df[cat_name], 0, 65536)
            for cat_name in clamp_list:
                df[cat_name] = functional.clamp(df[cat_name], 0, 16)
            for cat_name in firstx_list:
                df[cat_name + "firstx"] = functional.array_constructor(df[cat_name])
                df[cat_name + "firstx"] = functional.firstx(df[cat_name + "firstx"], 1)
            for cat_name in ngram_list:
                df[cat_name + "ngram"] = functional.array_constructor(df[cat_name])
                df[cat_name + "ngram"] = functional.onehot(df[cat_name + "ngram"], 0.0, 16.0, 8)
                df[cat_name + "ngram"] = functional.ngram(df[cat_name + "ngram"], 4)
                df[cat_name + "ngram"] = functional.firstx(df[cat_name + "ngram"], 1)

            # df["ngram"] = functional.array_constructor(df["cat_0"], df["cat_1"], df["cat_2"], df["cat_3"], df["cat_4"], df["cat_5"], df["cat_6"], df["cat_7"])
            # for i in range(13):
            #     df["ngram_{}".format(i)] = functional.ngram(df["ngram"], 4)
            #     df["ngram_{}".format(i)] = functional.firstx(df["ngram_{}".format(i)], 1)


        # # # flatten several columns into one
        df["dense_features"] = ta.dataframe(
            {int_name: df[int_name] for int_name in ["int_{}".format(i) for i in range(self.args.nDense)]}
        )
        sparse_name_list = ["cat_{}".format(i) for i in range(self.args.nSparse)]
        for cat_name in sparse_name_list:
            df[cat_name] = functional.array_constructor(df[cat_name])
        df["sparse_features"] = ta.dataframe(
            {cat_name: df[cat_name] for cat_name in sparse_name_list}
        )
        df = df[["dense_features", "sparse_features", DEFAULT_LABEL_NAME]]

        return df

    def preproc_2(self, df):
        for feature_name in DEFAULT_INT_NAMES:
            df[feature_name] = df[feature_name].fill_null(0)
            df[feature_name] = df[feature_name].cast(dt.float32)
        for feature_name in DEFAULT_CAT_NAMES:
            df[feature_name] = df[feature_name].fill_null(0)
            df[feature_name] = df[feature_name].cast(dt.int64)

        # ========================================================
        for i in range(self.args.nDense):
            df["int_{}".format(i)] = df["int_{}".format(i%13)]
        for i in range(self.args.nSparse):
            df["cat_{}".format(i)] = df["cat_{}".format(i%26)]

        logit_list = ["int_{}".format(i) for i in range(0,16)]
        boxcox_list = ["int_{}".format(i) for i in range(16,37)]
        onehot_list = ["int_{}".format(i) for i in range(37,52)]

        for i in range(4):
            for int_name in logit_list:
                df[int_name] = functional.logit(df[int_name], 1e-5)
            for int_name in boxcox_list:
                df[int_name] = functional.boxcox(df[int_name], 1e-5)
            for int_name in onehot_list:
                df[int_name + "_onehot"] = functional.array_constructor(df[int_name])
                df[int_name + "_onehot"] = functional.onehot(df[int_name + "_onehot"], 0.0, 16.0, 5)
                df[int_name + "_onehot"] = functional.firstx(df[int_name + "_onehot"], 1)

            sigrid_hash_list = ["cat_{}".format(i) for i in range(0,25)]
            clamp_list = ["cat_{}".format(i) for i in range(25,53)]
            firstx_list = ["cat_{}".format(i) for i in range(53,78)]
            ngram_list = ["cat_{}".format(i) for i in range(78,104)]

            for cat_name in sigrid_hash_list:
                df[cat_name] = functional.sigrid_hash(df[cat_name], 0, 65536)
            for cat_name in clamp_list:
                df[cat_name] = functional.clamp(df[cat_name], 0, 16)
            for cat_name in firstx_list:
                df[cat_name + "firstx"] = functional.array_constructor(df[cat_name])
                df[cat_name + "firstx"] = functional.firstx(df[cat_name + "firstx"], 1)
            for cat_name in ngram_list:
                df[cat_name + "ngram"] = functional.array_constructor(df[cat_name])
                df[cat_name + "ngram"] = functional.onehot(df[cat_name + "ngram"], 0.0, 16.0, 8)
                df[cat_name + "ngram"] = functional.ngram(df[cat_name + "ngram"], 4)
                df[cat_name + "ngram"] = functional.firstx(df[cat_name + "ngram"], 1)

            # df["ngram"] = functional.array_constructor(df["cat_0"], df["cat_1"], df["cat_2"], df["cat_3"], df["cat_4"], df["cat_5"], df["cat_6"], df["cat_7"])
            # for i in range(26):
            #     df["ngram_{}".format(i)] = functional.ngram(df["ngram"], 4)
            #     df["ngram_{}".format(i)] = functional.firstx(df["ngram_{}".format(i)], 1)

        # # # flatten several columns into one
        df["dense_features"] = ta.dataframe(
            {int_name: df[int_name] for int_name in ["int_{}".format(i) for i in range(self.args.nDense)]}
        )
        sparse_name_list = ["cat_{}".format(i) for i in range(self.args.nSparse)]
        for cat_name in sparse_name_list:
            df[cat_name] = functional.array_constructor(df[cat_name])
        df["sparse_features"] = ta.dataframe(
            {cat_name: df[cat_name] for cat_name in sparse_name_list}
        )
        df = df[["dense_features", "sparse_features", DEFAULT_LABEL_NAME]]

        return df


    def preprocess_wrap(self, raw_df, salt=0):
        if self.preprocessing_plan == 0 or self.preprocessing_plan == 1:
            return self.preproc(raw_df, salt)
        elif self.preprocessing_plan == 2:
            return self.preproc_1(raw_df)
        elif self.preprocessing_plan == 3:
            return self.preproc_2(raw_df)


    def criteo_collate(self, df):
        dense_features, kjt, labels = df.to_tensor(
            {
                # "dense_features": tap.rec.Dense(batch_first=True),
                "dense_features": _DenseTensorConversion(),
                "sparse_features": _JaggedTensorConversion(),
                "label": _Scalar(),
            }
        )

        return dense_features, kjt, labels

    def criteo_collate_skip_convert_kjt(self, df):
        dense_features, kjt, labels = df.to_tensor(
            {
                # "dense_features": tap.rec.Dense(batch_first=True),
                "dense_features": _DenseTensorConversion(),
                "sparse_features": _JaggedTensorConversionSkip(),
                "label": _Scalar(),
            }
        )

        return dense_features, self.kjt, labels
    
    def next(self):
        iter_dp = iter(self.parquet_df_dp)
        raw_df = next(iter_dp)
        # df = self.preproc(raw_df, salt=0)
        df = self.preprocess_wrap(raw_df, salt=0)
        preprocessing_result = self.criteo_collate(df)
        self.kjt = preprocessing_result[1]
        return preprocessing_result

    def next_skip_convert_kjt(self):
        iter_dp = iter(self.parquet_df_dp)
        raw_df = next(iter_dp)
        # # df = self.preproc(raw_df, salt=0)
        df = self.preprocess_wrap(raw_df, salt=0)

        dense_features, kjt, labels = self.criteo_collate_skip_convert_kjt(df)
        dense_features = dense_features.to(self.device)
        labels = labels.to(self.device).float()
        kjt = kjt.to(self.device)
        return dense_features, kjt, labels
    
    def next_skip_convert_kjt_cpu(self):
        iter_dp = iter(self.parquet_df_dp)
        raw_df = next(iter_dp)
        # # df = self.preproc(raw_df, salt=0)
        df = self.preprocess_wrap(raw_df, salt=0)

        dense_features, kjt, labels = self.criteo_collate_skip_convert_kjt(df)
        dense_features = dense_features
        labels = labels.float()
        kjt = kjt
        return dense_features, kjt, labels