
import os
import sys
sys.path.append('/workspace/RAP/RAP_end_to_end')

import argparse
from utils.input_splitter import table_mapping_processor, input_splitter
from utils.graph_parser import parse_file

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--preprocessing_plan', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--nDev', type=int, default=2)

op_list =  ["fill_null_dense", "fill_null_sparse", "sigrid_hash", "bucketize", "logit", "firstx", "boxcox", "clamp", "onehot", "ngram", "mapid"]

args = parser.parse_args()
file_name = "/workspace/RAP/RAP_end_to_end/preprocessing_plans/processing_plan_{}.txt".format(args.preprocessing_plan)
output_file_dir = "/workspace/RAP/RAP_end_to_end/splitted_input/"

parse_graph = parse_file(file_name) # draw_the_graph(parse_graph, args) # visualize the preprocessing graph
table_mp = table_mapping_processor(args.nDev, args.preprocessing_plan)

input_split = input_splitter(args.nDev, args.batch_size, args.preprocessing_plan, table_mp, parse_graph)
all_nodes_dense, all_nodes_sparse = input_split.split_input(output_file_dir)
placement = table_mp.get_table_placement()
print("finished splitting input!")