import re
import ast
import itertools
import graphviz
import argparse
import copy

parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
parser.add_argument(
    "--file_name", type=str, default=None, help="input_file_name"
)

class Data:
    new_id = itertools.count().__next__

    def __init__(self, content=None, predecessor=None):
        self.id = Data.new_id()
        self.content = content
        self.predecessor = predecessor
        self.successor = []
        self.dim = -1

    def add_parent(self, node):
        self.predecessor = node

    def add_child(self, node):
        self.successor.append(node)

class Node:
    new_id = itertools.count().__next__

    def __init__(self, op_name=None, raw_input=None, output_data=None, parameters=None):
        self.id = Node.new_id()

        self.predecessor = []
        self.successor = []
        self.sibling = []

        self.op_name = op_name

        self.input_data = []
        self.fill_null_index = -1
        self.raw_input = raw_input
        self.output_data = output_data

        self.parameters = parameters
        self.output_data.add_parent(self)
        self.layer = -1
        self.fused_place = None
        self.batch_size = None
        
    def add_parent(self, node):
        self.predecessor.append(node)
        
    def add_child(self, node):
        self.successor.append(node)

    def add_sibling(self, node):
        self.sibling.append(node)

    def add_input_data(self, data):
        self.input_data.append(data)
    
    def set_fused_place(self, gpu, layer, kernel_id, node_pls):
        self.fused_place = (gpu, layer, kernel_id, node_pls)

    def get_param(self):
        if self.op_name in ["fill_null", "sigrid_hash", "logit", "boxcox", "clamp", "mapid"]:
            return 1
        elif self.op_name in ["firstx", "onehot", "ngram"]:
            return int(self.parameters[-1])
        elif self.op_name == "bucketize":
            converted_list = ast.literal_eval(self.parameters[1])
            return len(converted_list)

    def get_width(self):
        if self.op_name == "fill_null":
            return 1
        else:
            return self.input_data[0].dim
    
    def get_out_width(self):
        if self.op_name == "fill_null":
            return 1
        else:
            return self.output_data.dim
        

class Graph:
    def __init__(self):
        self.nodes = []
        self.data = []
        self.variable = {}

        self.root_nodes = [] # where preprocessing starts
        self.leaf_nodes = [] # where preprocessing ends
        self.dense_feature_root_nodes = []
        self.sparse_feature_root_nodes = []
        self.dense_feature_leaf_nodes = []
        self.sparse_feature_leaf_nodes = []
        self.nNode = 0
        
    def add_node(self, new_node, input_data):
        for input in input_data:
            for node in reversed(self.nodes):
                if node.output_data.content == input:
                    node.add_child(new_node)
                    new_node.add_parent(node)
                    new_node.add_input_data(node.output_data)
                    break
            else:
                data = Data(input)
                new_node.add_input_data(data)
        self.nodes.append(new_node)
        self.nNode += 1

    def update_sibling(self):
        for node in self.nodes:
            for pre in node.predecessor:
                for suc in pre.successor:
                    if suc != node:
                        node.add_sibling(suc)

    def update_dim(self):
        for node in self.nodes:
            if not node.predecessor:
                node.input_data[0].dim = 1
            if node.op_name == "firstx":
                node.output_data.dim = int(node.parameters[1])
            elif node.op_name == "bucketize":
                converted_list = ast.literal_eval(node.parameters[1])
                node.output_data.dim = len(converted_list) + 1
            elif node.op_name == "ngram":
                width = node.input_data[0].dim
                n_of_gram = min(int(node.parameters[1]), width)
                node.output_data.dim = (width - n_of_gram + 1) * n_of_gram
            elif node.op_name == "onehot":
                node.output_data.dim = int(node.parameters[3])
            else:
                node.output_data.dim = node.input_data[0].dim

    def update_batch_size(self):
        for node in self.nodes:
            output_name = node.output_data.content
            if "int" in output_name: # dense feature
                node.batch_size = 1
            elif "cat" in output_name: # sparse feature
                node.batch_size = 2
            else:
                print("Error: output_name ERROR for updata_batch_size(): ", output_name)
                exit(0)

    def batch_size_trace_back_and_update(self):
        while True:
            cnt = 0
            for node in self.nodes:
                for idx, suc_node in enumerate(node.successor):
                    if node.batch_size != suc_node.batch_size:
                        node.successor.pop(idx)
                        id_node = Node(node.op_name, node.raw_input, node.output_data, node.parameters)
                        new_node = copy.deepcopy(node)
                        new_node.successor = [suc_node]
                        new_node.batch_size = suc_node.batch_size
                        new_node.id = id_node.id
                        self.nodes.append(new_node)
                        suc_node.predecessor.pop(0)
                        suc_node.predecessor.append(new_node)
                        print(new_node.id, new_node.op_name, new_node.parameters, new_node.batch_size, new_node.input_data[0].content, new_node.output_data.content, new_node.successor[0].op_name, len(new_node.successor))
                        cnt += 1
            if cnt == 0:
                break
    
    def updata_raw_date(self):
        for node in self.nodes:
           if node.op_name == "fill_null_dense" or node.op_name == "fill_null_sparse":
                prefix = "raw_"
                modified_input = node.input_data[0].content.replace(prefix, '')
                node.input_data[0].content = modified_input
                    
    def display(self):
        for node in self.nodes:
            print("=====================================")

            print(node.id, node.op_name, node.parameters)
            for data in node.input_data:
                print("     input_data: " + str(data.id), data.content)
            print("     output_data: " + str(node.output_data.id), node.output_data.content)

            for pre in node.predecessor:
                print("pre:", pre.id, pre.op_name, len(pre.input_data), pre.output_data.content, pre.parameters)
                for data in pre.input_data:
                    print("     input_data: " + str(data.id), data.content)
                print("     output_data: " + str(pre.output_data.id), pre.output_data.content)

            for suc in node.successor:
                print("suc:", suc.id, suc.op_name, suc.parameters)
                for data in suc.input_data:
                    print("     input_data: " + str(data.id), data.content)
                print("     output_data: " + str(suc.output_data.id), suc.output_data.content)

            for sib in node.sibling:
                print("sib:", sib.id, sib.op_name, len(sib.input_data), sib.output_data.content, sib.parameters)
                for data in sib.input_data:
                    print("     input_data: " + str(data.id), data.content)
                print("     output_data: " + str(sib.output_data.id), sib.output_data.content)

    def get_root_and_leaf(self):
        for node in self.nodes:
            if not node.predecessor:
                self.root_nodes.append(node)
            if not node.successor:
                self.leaf_nodes.append(node)

        for node in self.leaf_nodes:
            if "int" in node.output_data.content:
                self.dense_feature_leaf_nodes.append(node)
            else:
                self.sparse_feature_leaf_nodes.append(node)

        for node in self.root_nodes:
            if "int" in node.output_data.content:
                self.dense_feature_root_nodes.append(node)
            else:
                self.sparse_feature_root_nodes.append(node)

    def partition_graph_no_dup(self, nDev, table_mapping_dict):
        all_nodes_dense = []
        all_nodes_sparse = [[] for _ in range(nDev)]
       
        # =========================================
        # mapping node for dense feature
        visited_node = {}
        for node in self.dense_feature_leaf_nodes: 
            node_queue = [node]
            while len(node_queue) != 0:
                node = node_queue.pop(0)
                if node.id not in visited_node:
                    visited_node[node.id] = True
                    all_nodes_dense.append(node)
                    for pre in node.predecessor:
                        node_queue.append(pre)

        # print("all_nodes_dense: ", len(all_nodes_dense)) 

        # mapping node for sparse feature
        cnt = 0
        for node in self.sparse_feature_leaf_nodes:
            data_id = int(re.findall(r'\d+', node.output_data.content)[0])
            target_gpu = table_mapping_dict[data_id]

            node_queue = [node]
            visited_node = {}
            while len(node_queue) != 0:
                node = node_queue.pop(0)
                if node.id not in visited_node:
                    visited_node[node.id] = True
                    all_nodes_sparse[target_gpu].append(node)
                    cnt += 1
                    for pre in node.predecessor:
                        node_queue.append(pre)

        # print("all_nodes_sparse: ", cnt)
        # for i in range(nDev):
            # print("GPU:", i, len(all_nodes_sparse[i]))

        return all_nodes_dense, all_nodes_sparse

    def partition_graph_dup(self, nDev, table_mapping_dict):
        all_nodes_dense = []
        all_nodes_sparse = [[] for _ in range(nDev)]
       
        # =========================================
        # mapping node for dense feature
        for node in self.dense_feature_leaf_nodes: 
            node_queue = [node]
            while len(node_queue) != 0:
                node = node_queue.pop(0)
                all_nodes_dense.append(node)
                for pre in node.predecessor:
                    node_queue.append(pre)

        print("all_nodes_dense: ", len(all_nodes_dense)) 

        # mapping node for sparse feature
        cnt = 0
        for node in self.sparse_feature_leaf_nodes:
            data_id = int(re.findall(r'\d+', node.output_data.content)[0])
            target_gpu = table_mapping_dict[data_id]

            node_queue = [node]
            while len(node_queue) != 0:
                node = node_queue.pop(0)
                all_nodes_sparse[target_gpu].append(node)
                cnt += 1
                for pre in node.predecessor:
                    node_queue.append(pre)

        # print("all_nodes_sparse: ", cnt)
        for i in range(nDev):
            print("GPU:", i, len(all_nodes_sparse[i]))

        return all_nodes_dense, all_nodes_sparse

    
    def partition_dense_by_layer(self, nDev, nPartition):
        all_nodes_dense = []
        max_layer = 0        
        # =========================================
        # mapping node for dense feature
        visited_node = {}
        for node in self.dense_feature_leaf_nodes: 
            node_queue = [node]
            while len(node_queue) != 0:
                node = node_queue.pop(0)
                if node.id not in visited_node:
                    visited_node[node.id] = True
                    if len(node.successor) == 0:
                        node.layer = 0
                    else:
                        node.layer = node.successor[0].layer + 1
                        if node.layer > max_layer:
                            max_layer = node.layer
                    all_nodes_dense.append(node)
                    for pre in node.predecessor:
                        node_queue.append(pre)

        # print("all_nodes_dense: ", len(all_nodes_dense)) 

        layer_wise_partition = [[] for _ in range(nPartition)]
        for node in all_nodes_dense:
            layer_wise_partition[node.layer % nPartition].append(node)
        
        return layer_wise_partition

    def partition_dense_by_graph(self, nDev, nPartition):
        # self.dense_feature_leaf_nodes into nPartition chunks
        leaf_node_chunks = [[] for _ in range(nPartition)]
        for i in range(len(self.dense_feature_leaf_nodes)):
            leaf_node_chunks[i % nPartition].append(self.dense_feature_leaf_nodes[i])

        all_nodes_dense_list = []
        for j in range(nPartition):
            all_nodes_dense = []
            max_layer = 0        
            # =========================================
            # mapping node for dense feature
            visited_node = {}
            for node in leaf_node_chunks[j]: 
                node_queue = [node]
                while len(node_queue) != 0:
                    node = node_queue.pop(0)
                    if node.id not in visited_node:
                        visited_node[node.id] = True
                        if len(node.successor) == 0:
                            node.layer = 0
                        else:
                            node.layer = node.successor[0].layer + 1
                            if node.layer > max_layer:
                                max_layer = node.layer
                        all_nodes_dense.append(node)
                        for pre in node.predecessor:
                            node_queue.append(pre)
            all_nodes_dense_list.append(all_nodes_dense)

        return all_nodes_dense_list

    def partition_vertical(self, nDev, table_mapping_dict, nPartition):
        partition_nodes_dense = [[] for _ in range(nPartition)]
        partition_nodes_sparse = [[[] for _ in range(nPartition)] for _ in range(nDev)]

        trees_dense = []
        trees_sparse = [[] for _ in range(nDev)]

        # =========================================
        # partition dense nodes
        visited_node = {}
        for node in self.dense_feature_leaf_nodes:
            tree = []
            node_queue = [node]
            while len(node_queue) != 0:
                node = node_queue.pop(0)
                if node.id not in visited_node:
                    visited_node[node.id] = True
                    tree.append(node)
                    for pre in node.predecessor:
                        node_queue.append(pre)
            trees_dense.append(tree)

        trees_dense.sort(key=lambda x: len(x))

        # print("trees_dense: ", len(trees_dense))
        for tree in trees_dense:
            for node in tree:
                partition_nodes_dense[0].append(node)
            partition_nodes_dense.sort(key=lambda x: len(x))

        for i in range(nPartition):
            print(i, len(partition_nodes_dense[i]))

        for i in range(nPartition):
            for node in partition_nodes_dense[i]:
                print(i, node.id, node.op_name)

        # =========================================
        # partition sparse nodes
        for node in self.sparse_feature_leaf_nodes:
            data_id = int(re.findall(r'\d+', node.output_data.content)[0])
            target_gpu = table_mapping_dict[data_id]
            tree = []
            node_queue = [node]
            visited_node = {}
            while len(node_queue) != 0:
                node = node_queue.pop(0)
                if node.id not in visited_node:
                    visited_node[node.id] = True
                    tree.append(node)
                    for pre in node.predecessor:
                        node_queue.append(pre)
            trees_sparse[target_gpu].append(tree)

        for i in range(nDev):
            trees_sparse[i].sort(key=lambda x: len(x))

        # for i in range(nDev):
        #     print("trees_sparse: GPU_", i, len(trees_sparse[i]))

        for i in range(nDev):
            for tree in trees_sparse[i]:
                for node in tree:
                    partition_nodes_sparse[i][0].append(node)
                partition_nodes_sparse[i].sort(key=lambda x: len(x))

        # for i in range(nDev):
        #     for j in range(nPartition):
        #         print(i, j, len(partition_nodes_sparse[i][j]))
            
        return partition_nodes_dense, partition_nodes_sparse

    def partition_horizontal(self, nDev, table_mapping_dict, nPartition):
        partition_nodes_dense = [[] for _ in range(nPartition)]
        partition_nodes_sparse = [[[] for _ in range(nPartition)] for _ in range(nDev)]

        dense_nodes_queue = []
        dense_nodes = []

        sparse_nodes_queue = [[] for _ in range(nDev)]
        sparse_nodes = [[] for _ in range(nDev)]

        all_nodes_dense, all_nodes_sparse = self.partition_graph_no_dup(nDev, table_mapping_dict)

        # =========================================
        # partition dense nodes
        group = 0
        for node in reversed(self.dense_feature_leaf_nodes):
            dense_nodes_queue.append(node)

        visited_node = {}
        while dense_nodes_queue != []:
            node = dense_nodes_queue.pop(0)
            if node not in visited_node:
                dense_nodes.append(node)
                for pre in node.predecessor:
                    dense_nodes_queue.append(pre)

        dense_nodes = reversed(dense_nodes)

        for node in dense_nodes:
            partition_nodes_dense[group].append(node)
            
            if group < nPartition - 1 and len(partition_nodes_dense[group]) >= len(all_nodes_dense) // nPartition:
                group += 1

        # =========================================
        # partition sparse nodes
        groups = [0 for _ in range(nDev)]
        for node in reversed(self.sparse_feature_leaf_nodes):
            data_id = int(re.findall(r'\d+', node.output_data.content)[0])
            target_gpu = table_mapping_dict[data_id]
            sparse_nodes_queue[target_gpu].append(node)

        visited_node = {}
        for i in range(nDev):
            while sparse_nodes_queue[i] != []:
                node = sparse_nodes_queue[i].pop(0)
                if node not in visited_node:
                    sparse_nodes[i].append(node)
                    for pre in node.predecessor:
                        sparse_nodes_queue[i].append(pre)

        for i in range(nDev):
            sparse_nodes[i] = reversed(sparse_nodes[i])

        for i in range(nDev):
            for node in sparse_nodes[i]:
                partition_nodes_sparse[i][groups[i]].append(node)

                if groups[i] < nPartition - 1 and len(partition_nodes_sparse[i][groups[i]]) >= len(all_nodes_sparse[i]) // nPartition:
                    groups[i] += 1

        return partition_nodes_dense, partition_nodes_sparse
            

def parse_file(filename):
    graph = Graph()
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line[0] != '#':
                # get output_data
                output_data, node_operator = map(str.strip, line.split('='))

                if not output_data.startswith("df"):
                    graph.variable[output_data] = node_operator
                    print(graph.variable[output_data])
                    continue

                output_data = Data(output_data)

                # get op_name
                op_name = re.search(r"\.(\w+)\(", node_operator).group(1)

                # get input data
                input_data = re.findall("df\[\S+\]", node_operator)

                # get raw input
                raw_input = re.search(r"^[^\.]*", node_operator).group()
                if re.search("df\[\S+\]", raw_input) != None and re.fullmatch("df\[\S+\]", raw_input) == None:
                    pass
                else:
                    raw_input = None
                
                # seperate fillnull dense and sparse
                if op_name == "fill_null":
                    if "cat" in input_data[0]:
                        op_name += "_sparse"
                    else:
                        op_name += "_dense"
                # print(op_name, input_data, raw_input, "cat" in input_data[0])

                # get parameters
                parameters = []
                args = [arg.strip() for arg in re.search(r"\.(\w+)\((.*)\)", node_operator)[2].split(",")]
                i = 0
                while i < len(args):
                    if(args[i].startswith("[")):
                        merged_arg = args[i]
                        while True:
                            i += 1
                            merged_arg += ", " + args[i]
                            if(args[i].endswith("]")):
                                parameters.append(merged_arg)
                                break
                    else:
                        parameters.append(args[i])
                    i += 1

                for i in range(len(parameters)):
                    if graph.variable.get(parameters[i]) != None:
                        parameters[i] = graph.variable.get(parameters[i])

                # init node
                node = Node(op_name=op_name, raw_input=raw_input, output_data=output_data, parameters=parameters)
                graph.add_node(node, input_data)
    graph.update_sibling()
    graph.update_dim() 
    graph.get_root_and_leaf()
    graph.update_batch_size()
    graph.batch_size_trace_back_and_update()
    graph.updata_raw_date()

    return graph

if __name__ == "__main__":
    args = parser.parse_args()
    if args.file_name == None:
        file_name = "/workspace/RAP/preprocessing_graph_parser/preproc.txt"
    else:
        file_name = args.file_name
    parse_graph = parse_file(file_name)
    parse_graph.display()

    graph = graphviz.Digraph(comment="The Round Table")
    for node in parse_graph.nodes:
        # add node
        graph.attr('node', shape='ellipse', style='filled', color='lightblue')
        if node.raw_input != None:
            graph.node("node" + str(node.id), node.op_name + "\ninput: " + node.raw_input)
        else:
            graph.node("node" + str(node.id), node.op_name)

        param_content = "params:\n"
        for param in node.parameters:
            param_content += param + "\n"
        graph.edge("node" + str(node.id), "node" + str(node.id), label=param_content)

        # add input data and link to current node
        graph.attr('node', shape='diamond', style='filled', color='lightgrey')
        for data in node.input_data:
            graph.node("data" + str(data.id), data.content + "\ndim: " + str(data.dim))
            graph.edge("data" + str(data.id), "node" + str(node.id))
        
        # add output data and link to current node
        graph.node("data" + str(node.output_data.id), node.output_data.content + "\ndim: " + str(node.output_data.dim))
        graph.edge("node" + str(node.id), "data" + str(node.output_data.id))

    graph.format = 'png'
    graph.render("tree")

    for node in parse_graph.nodes:
        # if node.op_name == "ngram":
            # print(node.parameters)
        print(node.op_name, "param:", node.get_param(), "in_width:", node.get_width(), "out_width:", node.get_out_width())
    

