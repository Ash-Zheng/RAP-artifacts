# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This file pre-process the source file and save it as a TSV file and a Parquet file.
You do not need to re-run this file if "day_11_first_3k_rows.parquet" and "day_11_first_3k_rows.tsv" exist locally
"""

import pandas
import pyarrow
import pyarrow.parquet as parquet
from common import DEFAULT_CAT_NAMES, DEFAULT_COLUMN_NAMES, DEFAULT_INT_NAMES, safe_cast

file_name = "raw_data_4096" # raw_data_4096 raw_data_8192
input_name = "raw_data/{}.tsv".format(file_name)
# output_name = "parquet_data/{}.parquet".format(file_name)
output_name = "test_data_folder/{}.parquet".format(file_name)


# Read TSV File with Pandas
df = pandas.read_csv(input_name, sep="\t")
df.columns = DEFAULT_COLUMN_NAMES

# Convert hex strings to interger
for i, row in df.iterrows():
    for cat_col in DEFAULT_CAT_NAMES:
        df.at[i, cat_col] = safe_cast(row[cat_col], int, 0)

    for int_col in DEFAULT_INT_NAMES:
        df.at[i, int_col] = safe_cast(row[int_col], int, 0)

# Convert to PyArrow table and write to disk as parquet file
table = pyarrow.Table.from_pandas(df=df)
parquet.write_table(table, output_name)

# Write to a new .tsv file
# df.to_csv("day_11_first_3k_rows.tsv", sep="\t")