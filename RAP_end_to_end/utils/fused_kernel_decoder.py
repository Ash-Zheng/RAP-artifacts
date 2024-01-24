import re

class fused_kernel_node:
    def __init__(self, kernel_list, all_nodes):
        self.op_name = None if len(kernel_list) == 0 else kernel_list[0][0][:-2]
        self.node_list = []
        self.width_list = []
        self.param_list= []
        self.nOp = 0
        self.global_id = -1

        self.input_tensor_list_name_list = []
        self.output_tensor_list_name_list = []


        self.batch_size = None if len(kernel_list) == 0 else kernel_list[0][0][-1:]
        if self.batch_size == "1":
            self.data_size = "args.batch_size"
        elif self.batch_size == "2":
            self.data_size = "args.batch_size * args.nDev"

        for node_info in kernel_list:
            node_id = node_info[1]
            node = all_nodes[node_id]
            self.node_list.append(node_id)
            self.width_list.append(node.get_width())
            self.param_list.append(node.get_param())
            self.nOp += 1

        # code gen parameters
        self.input_argument_list = []
        self.output_argument_list = []
        self.input_tensor_list_name = None
        self.output_tensor_list_name = None
        self.input_ptr_name = None
        self.input_length_name = None
        self.data_prepare_code = [] # execute only once
        self.input_list_code = [] # execute each iteration
        self.cpu_part_code = [] # execute each iteration
        self.gpu_part_code = [] # execute each iteration

        self.data_prepare_code_next = [] # execute only once
        self.input_list_code_next = [] # execute each iteration
        self.cpu_part_code_next = [] # execute each iteration
        self.gpu_part_code_next = [] # execute each iteration

        self.gpu_kernel_arg_list = []
        self.gpu_kernel_arg_list_next = []



    def kernel_shard(self, shard_nOp):
        kernel_1 = fused_kernel_node([], None)
        kernel_2 = fused_kernel_node([], None)
        # Kernel-1
        kernel_1.op_name = self.op_name
        kernel_1.batch_size = self.batch_size
        kernel_1.data_size = self.data_size
        print("shard_nOp:{}, nop:{}".format(shard_nOp, self.nOp))
        for i in range(shard_nOp):
            kernel_1.node_list.append(self.node_list[i])
            kernel_1.width_list.append(self.width_list[i])
            kernel_1.param_list.append(self.param_list[i])
            kernel_1.nOp += 1
        # Kernel-2
        kernel_2.op_name = self.op_name
        kernel_2.batch_size = self.batch_size
        kernel_2.data_size = self.data_size
        for i in range(shard_nOp, self.nOp):
            kernel_2.node_list.append(self.node_list[i])
            kernel_2.width_list.append(self.width_list[i])
            kernel_2.param_list.append(self.param_list[i])
            kernel_2.nOp += 1
        
        return kernel_1, kernel_2
    
    def gen_code(self, file_name):
        with open(file_name, "a") as f:
            f.write("# {}-{} input_list_code:\n".format(self.op_name, self.global_id))
            for line in self.input_list_code:
                f.write(line + "\n")
            f.write("\n")
            f.write("# {}-{} data_prepare_code:\n".format(self.op_name, self.global_id))
            for line in self.data_prepare_code:
                f.write(line + "\n")
            f.write("\n")
            f.write("# {}-{} cpu_part_code:\n".format(self.op_name, self.global_id))
            for line in self.cpu_part_code:
                f.write(line + "\n")
            f.write("\n")
            f.write("# {}-{} gpu_part_code:\n".format(self.op_name, self.global_id))
            for line in self.gpu_part_code:
                f.write(line + "\n")
            f.write("\n")

    def print_codes(self, codes_type):
        if codes_type == "data_prepare_code":
            codes = self.data_prepare_code
        elif codes_type == "cpu_part_code":
            codes = self.cpu_part_code
        elif codes_type == "gpu_part_code":
            codes = self.gpu_part_code
        elif codes_type == "input_list_code":
            codes = self.input_list_code
        else:
            print("ERROR: code type error! type get:{}".format(codes_type))
            exit(1)

        print("# {}-{} {}:".format(self.op_name, self.global_id, codes_type))
        for line in codes:
            print(line)
        print("")

    def get_predecessor_data_size(self, all_node, schedule):
        if len(self.node_list) > 0:
            this_node = all_node[self.node_list[0]]
            gpu, layer, kernel_id, _ = this_node.predecessor[0].fused_place
            return schedule[gpu][layer][kernel_id].data_size
        else:
            print("ERROR: empty predecessor node")
            exit(-1)

    def init_input_name(self, all_node, schedule=None): # execute only once (allocate input)
        if self.op_name == "fill_null_dense":
            self.input_tensor_list_name = self.op_name + "_ptr_list_{}".format(self.global_id)
            self.output_tensor_list_name = self.op_name + "_output_tensor_list_{}".format(self.global_id)

            self.input_tensor_list_name_list.append(self.input_tensor_list_name)
            self.output_tensor_list_name_list.append(self.output_tensor_list_name)
            # self.data_size = "args.batch_size"
        elif self.op_name == "fill_null_sparse":
            self.input_tensor_list_name = self.op_name + "_ptr_list_{}".format(self.global_id)
            self.output_tensor_list_name = self.op_name + "_output_tensor_list_{}".format(self.global_id)

            self.input_tensor_list_name_list.append(self.input_tensor_list_name)
            self.output_tensor_list_name_list.append(self.output_tensor_list_name)
            # self.data_size = "args.batch_size * nDev"
        elif self.op_name in ["logit", "sigrid_hash", "boxcox", "clamp", "mapid"]:
            self.input_tensor_list_name = self.op_name + "_input_list_{}".format(self.global_id)
            self.output_tensor_list_name = self.input_tensor_list_name

            self.input_tensor_list_name_list.append(self.input_tensor_list_name)
            self.output_tensor_list_name_list.append(self.output_tensor_list_name)
            # self.data_size = self.get_predecessor_data_size(all_node, schedule)
        elif self.op_name in ["firstx", "bucketize", "ngram", "onehot"]:
            self.input_tensor_list_name = self.op_name + "_input_list_{}".format(self.global_id)
            self.output_tensor_list_name = self.op_name + "_output_list_{}".format(self.global_id)

            self.input_tensor_list_name_list.append(self.input_tensor_list_name)
            self.output_tensor_list_name_list.append(self.output_tensor_list_name)
            # self.data_size = self.get_predecessor_data_size(all_node, schedule)
        else:
            print("ERROR: Unknow op_name:{}".format(self.op_name))
            exit(-1)

    def init_input_list_code(self, all_node, schedule):
        if self.op_name == "fill_null_dense":
            # line 1: get ptrs
            line_1 = " = ["
            for node_id in self.node_list:
                this_node = all_node[node_id]
                data_name = this_node.input_data[0].content
                query_name = data_name.split("\"")[1]
                if self.batch_size == "1":
                    line_1 += "df_dense[\"{}\"].data.ptr, ".format(query_name)
                elif self.batch_size == "2":
                    line_1 += "df_sparse[\"{}\"].data.ptr, ".format(query_name)
            line_1 = line_1[:-2]
            line_1 += "]"

            line_1_next = " = ["
            for node_id in self.node_list:
                this_node = all_node[node_id]
                data_name = this_node.input_data[0].content
                query_name = data_name.split("\"")[1]
                if self.batch_size == "1":
                    line_1_next += "df_dense_next[\"{}\"].data.ptr, ".format(query_name)
                elif self.batch_size == "2":
                    line_1_next += "df_sparse_next[\"{}\"].data.ptr, ".format(query_name)
            line_1_next = line_1_next[:-2]
            line_1_next += "]"
            self.input_list_code.append(self.input_tensor_list_name + line_1)
            self.input_list_code_next.append(self.input_tensor_list_name + "_next" + line_1_next)

        elif self.op_name == "fill_null_sparse":
            # line 1: get ptrs
            # line_1 = self.input_tensor_list_name + " = ["
            line_1 = " = ["
            for node_id in self.node_list:
                this_node = all_node[node_id]
                data_name = this_node.input_data[0].content
                query_name = data_name.split("\"")[1]
                line_1 += "df_sparse[\"{}\"].data.ptr, ".format(query_name)
            line_1 = line_1[:-2]
            line_1 += "]"
            # self.input_list_code.append(line_1)
            self.input_list_code.append(self.input_tensor_list_name + line_1)

            line_1_next = " = ["
            for node_id in self.node_list:
                this_node = all_node[node_id]
                data_name = this_node.input_data[0].content
                query_name = data_name.split("\"")[1]
                line_1_next += "df_sparse_next[\"{}\"].data.ptr, ".format(query_name)
            line_1_next = line_1_next[:-2]
            line_1_next += "]"
            self.input_list_code_next.append(self.input_tensor_list_name + "_next" + line_1_next)

        elif self.op_name in ["logit", "sigrid_hash", "firstx", "bucketize", "ngram", "onehot", "boxcox", "clamp", "mapid"]:
            # line 1: prepare input list ()
            # line_1 = "{} = [".format(self.input_tensor_list_name)
            line_1 = " = ["
            for node_id in self.node_list:
                this_node = all_node[node_id]
                gpu, layer, kernel_id, node_place = this_node.predecessor[0].fused_place
                input_data_name = schedule[gpu][layer][kernel_id].output_tensor_list_name
                # if schedule[gpu][layer][kernel_id].op_name == "fill_null_dense" or schedule[gpu][layer][kernel_id].op_name == "fill_null_sparse":
                    # node_place = schedule[gpu][layer][kernel_id].fill_null_index
                line_1 += "{}[{}], ".format(input_data_name, node_place)
            line_1 = line_1[:-2]
            line_1 += "]"

            line_1_next = " = ["
            for node_id in self.node_list:
                this_node = all_node[node_id]
                gpu, layer, kernel_id, node_place = this_node.predecessor[0].fused_place
                input_data_name = schedule[gpu][layer][kernel_id].output_tensor_list_name
                line_1_next += "{}_next[{}], ".format(input_data_name, node_place)
            line_1_next = line_1_next[:-2]
            line_1_next += "]"
            # self.input_list_code.append(line_1)
            self.input_list_code.append(self.input_tensor_list_name + line_1)

            self.input_list_code_next.append(self.input_tensor_list_name + "_next" + line_1_next)

    def init_data_prepare_code(self, all_node=None): # execute only once (allocate space)
        if self.op_name == "fill_null_dense":
            # line 1: allocate ouput tensor
            line_1 = "{} = [torch.zeros(({}, 1), dtype=torch.float, device=device) for _ in range(len({}))]".format(
                self.output_tensor_list_name,
                self.data_size,
                self.input_tensor_list_name,
            ) # allocate output tensor buffer
            self.data_prepare_code.append(line_1)
            line_1_next = "{}_next = [torch.zeros(({}, 1), dtype=torch.float, device=device) for _ in range(len({}_next))]".format(
                self.output_tensor_list_name,
                self.data_size,
                self.input_tensor_list_name,
            ) # allocate output tensor buffer
            self.data_prepare_code_next.append(line_1_next)

            # line 2: get ouput tensor ptr
            out_ptr_name = self.output_tensor_list_name + "_ptr"
            line_2 = "{} = cuda_preprocess.copy_tensor_list_to_GPU_tensor({})".format(
                out_ptr_name,
                self.output_tensor_list_name
            ) # create the pointer to the output tensor buffer
            self.data_prepare_code.append(line_2)
            line_2_next = "{}_next = cuda_preprocess.copy_tensor_list_to_GPU_tensor({}_next)".format(
                out_ptr_name,
                self.output_tensor_list_name
            ) # create the pointer to the output tensor buffer
            self.data_prepare_code_next.append(line_2_next)
            
        elif self.op_name == "fill_null_sparse":
            # line 1: allocate ouput tensor
            line_1 = "{} = [torch.zeros(({}, 1), dtype=torch.int64, device=device) for _ in range(len({}))]".format(
                self.output_tensor_list_name,
                self.data_size,
                self.input_tensor_list_name,
                )
            self.data_prepare_code.append(line_1)
            line_1_next = "{}_next = [torch.zeros(({}, 1), dtype=torch.int64, device=device) for _ in range(len({}_next))]".format(
                self.output_tensor_list_name,
                self.data_size,
                self.input_tensor_list_name,
            )
            self.data_prepare_code_next.append(line_1_next)
            
            # line 2: get ouput tensor ptr
            out_ptr_name = self.output_tensor_list_name + "_ptr"
            line_2 = "{} = cuda_preprocess.copy_tensor_list_to_GPU_tensor({})".format(
                out_ptr_name,
                self.output_tensor_list_name
            ) # create the pointer to the output tensor buffer
            self.data_prepare_code.append(line_2)
            line_2_next = "{}_next = cuda_preprocess.copy_tensor_list_to_GPU_tensor({}_next)".format(
                out_ptr_name,
                self.output_tensor_list_name
            )
            self.data_prepare_code_next.append(line_2_next)
        
        elif self.op_name == "logit":
            # line 1: get eps_list:
            line_1 = "logit_eps_list_{} = [".format(self.global_id)
            for node_id in self.node_list:
                this_node = all_node[node_id]
                line_1 += this_node.parameters[1]
                line_1 += ", "
            line_1 = line_1[:-2]
            line_1 += "]"
            self.data_prepare_code.append(line_1)
            self.data_prepare_code_next.append(line_1)

            # line 2: convert eps list to gpu ptr
            line_2 = "logit_eps_list_ptr_{} = cuda_preprocess.logit_ptr_prepare({}, logit_eps_list_{})".format(self.global_id, self.input_tensor_list_name, self.global_id)
            self.data_prepare_code.append(line_2)
            line_2_next = "logit_eps_list_ptr_{} = cuda_preprocess.logit_ptr_prepare({}_next, logit_eps_list_{})".format(self.global_id, self.input_tensor_list_name, self.global_id)
            self.data_prepare_code_next.append(line_2_next)

        elif self.op_name == "sigrid_hash":
            # line 1: get hash length list:
            line_1 = "sigrid_hash_length_list_{} = [".format(self.global_id)
            for node_id in self.node_list:
                this_node = all_node[node_id]
                line_1 += this_node.parameters[-1]
                line_1 += ", "
            line_1 = line_1[:-2]
            line_1 += "]"
            self.data_prepare_code.append(line_1)
            self.data_prepare_code_next.append(line_1)

            # line 2: convert hash length to gpu ptr
            line_2 = "gpu_table_list_{}, gpu_multiplier_list_{}, gpu_shift_list_{} = cuda_preprocess.sigridhash_ptr_prepare(sigrid_hash_length_list_{})".format(self.global_id, self.global_id, self.global_id, self.global_id)
            self.data_prepare_code.append(line_2)
            self.data_prepare_code_next.append(line_2)

        elif self.op_name == "firstx":
            # line 1: get x list:
            line_1 = "x_list_{} = [".format(self.global_id)
            for node_id in self.node_list:
                this_node = all_node[node_id]
                line_1 += this_node.parameters[-1]
                line_1 += ", "
            line_1 = line_1[:-2]
            line_1 += "]"
            self.data_prepare_code.append(line_1)
            self.data_prepare_code_next.append(line_1)

            # line 2: convert x list to gpu ptr
            line_2 = "x_list_ptr_{}, firstx_width_list_ptr_{} = cuda_preprocess.firstx_ptr_prepare({}, x_list_{})".format(
                self.global_id, 
                self.global_id, 
                self.input_tensor_list_name, 
                self.global_id
            )
            line_2_next = "x_list_ptr_{}, firstx_width_list_ptr_{} = cuda_preprocess.firstx_ptr_prepare({}_next, x_list_{})".format(
                self.global_id, 
                self.global_id, 
                self.input_tensor_list_name, 
                self.global_id
            )
            self.data_prepare_code.append(line_2)
            self.data_prepare_code_next.append(line_2_next)
        
        elif self.op_name == "bucketize":
            # line 1: get border list:
            line_1 = "border_list_{} = [".format(self.global_id)
            for node_id in self.node_list:
                this_node = all_node[node_id]
                border_list = this_node.parameters[-1]
                tensor_alloc = "torch.tensor({}, dtype=torch.float32, device=device)".format(border_list)
                line_1 += tensor_alloc
                line_1 += ", "
            line_1 = line_1[:-2]
            line_1 += "]"
            self.data_prepare_code.append(line_1)
            self.data_prepare_code_next.append(line_1)

            # line 2: convert border list to gpu ptr
            line_2 = "bucketize_border_list_ptr_{}, bucketize_length_list_ptr_{} = cuda_preprocess.bucketize_ptr_prepare({}, border_list_{})".format(
                self.global_id,
                self.global_id,
                self.input_tensor_list_name, 
                self.global_id
            )
            line_2_next = "bucketize_border_list_ptr_{}, bucketize_length_list_ptr_{} = cuda_preprocess.bucketize_ptr_prepare({}_next, border_list_{})".format(
                self.global_id,
                self.global_id,
                self.input_tensor_list_name, 
                self.global_id
            )
            self.data_prepare_code.append(line_2)
            self.data_prepare_code_next.append(line_2_next)
        
        elif self.op_name == "boxcox":
            # line 1: get lambda_list:
            line_1 = "boxcox_lambda_list_{} = [".format(self.global_id)
            for node_id in self.node_list:
                this_node = all_node[node_id]
                line_1 += this_node.parameters[1]
                line_1 += ", "
            line_1 = line_1[:-2]
            line_1 += "]"
            self.data_prepare_code.append(line_1)

            # line 2: convert lambda list to gpu ptr
            line_2 = "lambda_list_ptr_{} = cuda_preprocess.boxcox_ptr_prepare({}, boxcox_lambda_list_{})".format(self.global_id, self.input_tensor_list_name, self.global_id)
            self.data_prepare_code.append(line_2)
        
        elif self.op_name == "clamp":
            # line 1 and line 2: get min and max list:
            line_1 = "clamp_min_list_{} = [".format(self.global_id)
            line_2 = "clamp_max_list_{} = [".format(self.global_id)

            for node_id in self.node_list:
                this_node = all_node[node_id]
                line_1 += this_node.parameters[1]
                line_1 += ", "
                line_2 += this_node.parameters[2]
                line_2 += ", "
            line_1 = line_1[:-2]
            line_1 += "]"
            line_2 = line_2[:-2]
            line_2 += "]"
            self.data_prepare_code.append(line_1)
            self.data_prepare_code.append(line_2)

            # line 3: convert min and max list to gpu ptr
            # min_list_ptr, max_list_ptr = cuda_preprocess.clamp_ptr_prepare(input_list, min_list, max_list)
            line_3 = "clamp_min_list_ptr_{}, clamp_max_list_ptr_{} = cuda_preprocess.clamp_ptr_prepare({}, clamp_min_list_{}, clamp_max_list_{})".format(self.global_id, self.global_id, self.input_tensor_list_name, self.global_id, self.global_id)
            self.data_prepare_code.append(line_3)

        elif self.op_name == "ngram":
            # line 1: get n_list:
            line_1 = "ngram_n_list_{} = [".format(self.global_id)
            for node_id in self.node_list:
                this_node = all_node[node_id]
                line_1 += this_node.parameters[1]
                line_1 += ", "
            line_1 = line_1[:-2]
            line_1 += "]"
            self.data_prepare_code.append(line_1)

            # line 2: convert ngram list to gpu ptr
            # width_ptr, n_list_ptr = cuda_preprocess.ngram_ptr_prepare(input_list, n_list)
            line_2 = "ngram_width_ptr_{}, ngram_n_list_ptr_{} = cuda_preprocess.ngram_ptr_prepare({}, ngram_n_list_{})".format(self.global_id, self.global_id, self.input_tensor_list_name, self.global_id)
            self.data_prepare_code.append(line_2)

        elif self.op_name == "onehot":
            # line 1 2 3: get input list:
            line_1 = "onehot_low_list_{} = [".format(self.global_id)
            line_2 = "onehot_high_list_{} = [".format(self.global_id)
            line_3 = "onehot_num_classes_list_{} = [".format(self.global_id)
            for node_id in self.node_list:
                this_node = all_node[node_id]
                line_1 += this_node.parameters[1]
                line_1 += ", "
                line_2 += this_node.parameters[2]
                line_2 += ", "
                line_3 += this_node.parameters[3]
                line_3 += ", "
            line_1 = line_1[:-2]
            line_1 += "]"
            line_2 = line_2[:-2]
            line_2 += "]"
            line_3 = line_3[:-2]
            line_3 += "]"
            self.data_prepare_code.append(line_1)
            self.data_prepare_code.append(line_2)
            self.data_prepare_code.append(line_3)

            # line 4: convert onehot list to gpu ptr
            line_4 = "onehot_low_list_ptr_{}, onehot_high_list_ptr_{}, onehot_num_classes_list_ptr_{} = cuda_preprocess.onehot_ptr_prepare({}, onehot_low_list_{}, onehot_high_list_{}, onehot_num_classes_list_{})".format(self.global_id, self.global_id, self.global_id, self.input_tensor_list_name, self.global_id, self.global_id, self.global_id)
            self.data_prepare_code.append(line_4)


    def init_cpu_part_code(self, all_node, schedule=None): # execute each iteration (prepare data ptr)
        if self.op_name in ["fill_null_dense", "fill_null_sparse"]:
            # line 1: convert ptr to tensor
            input_tensor_name = self.input_tensor_list_name + "_tensor"
            line_1 = "{} = torch.tensor({}, dtype=torch.int64, device=device)".format(input_tensor_name, self.input_tensor_list_name)
            line_1_next = "{}_next = torch.tensor({}_next, dtype=torch.int64, device=device)".format(input_tensor_name, self.input_tensor_list_name)
            self.cpu_part_code.append(line_1)
            self.cpu_part_code_next.append(line_1_next)

        elif self.op_name == "sigrid_hash":
            # line 1: convert tensor to ptr
            self.input_ptr_name = self.op_name + "_input_ptr_{}".format(self.global_id)
            self.input_length_name = self.op_name + "_input_length_{}".format(self.global_id)
            offset_name = self.op_name + "_offset_length_{}".format(self.global_id)
            line_1 = "{}, {}, {} = cuda_preprocess.sigridhash_cpu_part_base({})".format(self.input_ptr_name, offset_name, self.input_length_name, self.input_tensor_list_name)
            line_1_next = "{}_next, {}_next, {}_next = cuda_preprocess.sigridhash_cpu_part_base({}_next)".format(self.input_ptr_name, offset_name, self.input_length_name, self.input_tensor_list_name)
            self.cpu_part_code.append(line_1)
            self.cpu_part_code_next.append(line_1_next)

        elif self.op_name == "firstx":
            # line 1: generate output tensor and convert tensor to ptr
            self.input_ptr_name = self.op_name + "_input_ptr_{}".format(self.global_id)
            self.input_length_name = self.op_name + "_input_length_{}".format(self.global_id)
            line_1 = "{}, firstx_out_ptr_{}, {}, {} = cuda_preprocess.firstx_cpu_part({}, x_list_{})".format(
                self.input_ptr_name, 
                self.global_id, 
                self.input_length_name, 
                self.output_tensor_list_name,
                self.input_tensor_list_name,
                self.global_id
            )
            line_1_next = "{}_next, firstx_out_ptr_{}_next, {}_next, {}_next = cuda_preprocess.firstx_cpu_part({}_next, x_list_{})".format(
                self.input_ptr_name, 
                self.global_id, 
                self.input_length_name, 
                self.output_tensor_list_name,
                self.input_tensor_list_name,
                self.global_id
            )
            self.cpu_part_code.append(line_1)
            self.cpu_part_code_next.append(line_1_next)

        elif self.op_name == "bucketize":
            # line 1: generate output tensor and convert tensor to ptr
            self.input_ptr_name = self.op_name + "_input_ptr_{}".format(self.global_id)
            self.input_length_name = self.op_name + "_input_length_{}".format(self.global_id)
            line_1 = "{}, bucketize_out_ptr_{}, {}, {} = cuda_preprocess.bucketize_cpu_part({}, border_list_{})".format(
                self.input_ptr_name, 
                self.global_id, 
                self.input_length_name, 
                self.output_tensor_list_name,
                self.input_tensor_list_name,
                self.global_id
            )
            line_1_next = "{}_next, bucketize_out_ptr_{}_next, {}_next, {}_next = cuda_preprocess.bucketize_cpu_part({}_next, border_list_{})".format(
                self.input_ptr_name, 
                self.global_id, 
                self.input_length_name, 
                self.output_tensor_list_name,
                self.input_tensor_list_name,
                self.global_id
            )
            self.cpu_part_code.append(line_1)
            self.cpu_part_code_next.append(line_1_next)

        elif self.op_name in ["boxcox", "logit", "clamp"]:
            # line 1: convert tensor to ptr
            self.input_ptr_name = self.op_name + "_input_ptr_{}".format(self.global_id)
            self.input_length_name = self.op_name + "_input_length_{}".format(self.global_id)
            line_1 = "{}, {} = cuda_preprocess.{}_cpu_part_base({})".format(self.input_ptr_name, self.input_length_name, self.op_name, self.input_tensor_list_name)
            self.cpu_part_code.append(line_1)
        
        elif self.op_name == "ngram":
            # line 1: convert tensor to ptr
            self.input_ptr_name = self.op_name + "_input_ptr_{}".format(self.global_id)
            self.input_length_name = self.op_name + "_input_length_{}".format(self.global_id)
            line_1 = "{}, ngram_out_ptr_{}, {}, {} = cuda_preprocess.ngram_cpu_part({}, ngram_n_list_{})".format(
                self.input_ptr_name, 
                self.global_id, 
                self.input_length_name,
                self.output_tensor_list_name,
                self.input_tensor_list_name,
                self.global_id
            )
            self.cpu_part_code.append(line_1)
        
        elif self.op_name == "onehot":
            # tensor_list_ptr, tensor_out_list_ptr, total_length, result = cuda_preprocess.onehot_cpu_part(input_list[0:upper_bound], num_classes_list[0:upper_bound])
            # line 1: convert tensor to ptr
            self.input_ptr_name = self.op_name + "_input_ptr_{}".format(self.global_id)
            self.input_length_name = self.op_name + "_input_length_{}".format(self.global_id)
            line_1 = "{}, onehot_out_ptr_{}, {}, {} = cuda_preprocess.onehot_cpu_part({}, onehot_num_classes_list_{})".format(
                self.input_ptr_name, 
                self.global_id, 
                self.input_length_name,
                self.output_tensor_list_name,
                self.input_tensor_list_name,
                self.global_id
            )
            self.cpu_part_code.append(line_1)

    def init_gpu_part_code(self): # execute each iteration (prepare data ptr)
        if self.op_name == "fill_null_dense":
            # line 1: launch fill null float kernel
            input_tensor_name = self.input_tensor_list_name + "_tensor"
            out_ptr_name = self.output_tensor_list_name + "_ptr"
            line_1 = "cuda_preprocess.fill_null_float_list_gpu_part_tensor({}, {}, {})".format(input_tensor_name, out_ptr_name, self.data_size)
            line_1_next = "cuda_preprocess.fill_null_float_list_gpu_part_tensor({}_next, {}_next, {})".format(input_tensor_name, out_ptr_name, self.data_size)
            self.gpu_part_code.append(line_1)
            self.gpu_part_code_next.append(line_1_next)
            self.gpu_kernel_arg_list.extend([input_tensor_name, out_ptr_name])
            self.gpu_kernel_arg_list_next.extend([input_tensor_name+"_next", out_ptr_name+"_next"])

        elif self.op_name == "fill_null_sparse":
            # line 1: launch fill null int64 kernel
            input_tensor_name = self.input_tensor_list_name + "_tensor"
            out_ptr_name = self.output_tensor_list_name + "_ptr"
            line_1 = "cuda_preprocess.fill_null_int64_list_gpu_part_tensor({}, {}, {})".format(input_tensor_name, out_ptr_name, self.data_size)
            line_1_next = "cuda_preprocess.fill_null_int64_list_gpu_part_tensor({}_next, {}_next, {})".format(input_tensor_name, out_ptr_name, self.data_size)
            self.gpu_part_code.append(line_1)
            self.gpu_part_code_next.append(line_1_next)
            self.gpu_kernel_arg_list.extend([input_tensor_name, out_ptr_name])
            self.gpu_kernel_arg_list_next.extend([input_tensor_name+"_next", out_ptr_name+"_next"])

        elif self.op_name == "logit":
            # line 1: launch logit kernel
            line_1 = "cuda_preprocess.logit_gpu_part({}, logit_eps_list_ptr_{}, {}, {})".format(
                self.input_ptr_name, 
                self.global_id, 
                self.input_length_name, 
                self.data_size
            )
            line_1_next = "cuda_preprocess.logit_gpu_part({}_next, logit_eps_list_ptr_{}, {}_next, {})".format(
                self.input_ptr_name, 
                self.global_id, 
                self.input_length_name, 
                self.data_size
            )
            self.gpu_part_code.append(line_1)
            self.gpu_part_code_next.append(line_1_next)
            self.gpu_kernel_arg_list.extend([self.input_ptr_name, "logit_eps_list_ptr_{}".format(self.global_id), self.input_length_name])
            self.gpu_kernel_arg_list_next.extend([self.input_ptr_name+"_next", "logit_eps_list_ptr_{}".format(self.global_id), self.input_length_name+"_next"])

        elif self.op_name == "sigrid_hash":
            # line 1: launch sigrid hash kernel
            offset_name = self.op_name + "_offset_length_{}".format(self.global_id)
            line_1 = "cuda_preprocess.sigridhash_gpu_part({}, {}, 0, gpu_table_list_{}, gpu_multiplier_list_{}, gpu_shift_list_{}, {}, {})".format(
                self.input_ptr_name,
                offset_name, 
                self.global_id, 
                self.global_id, 
                self.global_id, 
                self.input_length_name, 
                self.data_size
            )
            line_1_next = "cuda_preprocess.sigridhash_gpu_part({}_next, {}_next, 0, gpu_table_list_{}, gpu_multiplier_list_{}, gpu_shift_list_{}, {}_next, {})".format(
                self.input_ptr_name,
                offset_name, 
                self.global_id, 
                self.global_id, 
                self.global_id, 
                self.input_length_name, 
                self.data_size
            )
            self.gpu_part_code.append(line_1)
            self.gpu_part_code_next.append(line_1_next)
            self.gpu_kernel_arg_list.extend([self.input_ptr_name, offset_name, "gpu_table_list_{}".format(self.global_id), "gpu_multiplier_list_{}".format(self.global_id), "gpu_shift_list_{}".format(self.global_id), self.input_length_name])
            self.gpu_kernel_arg_list_next.extend([self.input_ptr_name+"_next", offset_name+"_next", "gpu_table_list_{}".format(self.global_id), "gpu_multiplier_list_{}".format(self.global_id), "gpu_shift_list_{}".format(self.global_id), self.input_length_name+"_next"])

        elif self.op_name == "firstx":
            # line 1: launch firstx kernel
            if_float = "1" if self.data_size == "args.batch_size" else "0" # dense -> float-firstx, sparse -> int64-firstx
            line_1 = "cuda_preprocess.firstx_gpu_part({}, firstx_out_ptr_{}, x_list_ptr_{}, firstx_width_list_ptr_{}, {}, {}, {})".format(
                self.input_ptr_name, 
                self.global_id, 
                self.global_id, 
                self.global_id, 
                self.input_length_name, 
                self.data_size, 
                if_float
            )
            line_1_next = "cuda_preprocess.firstx_gpu_part({}_next, firstx_out_ptr_{}_next, x_list_ptr_{}, firstx_width_list_ptr_{}, {}_next, {}, {})".format(
                self.input_ptr_name, 
                self.global_id, 
                self.global_id, 
                self.global_id, 
                self.input_length_name, 
                self.data_size, 
                if_float
            )
            self.gpu_part_code.append(line_1)
            self.gpu_part_code_next.append(line_1_next)
            self.gpu_kernel_arg_list.extend([self.input_ptr_name, "firstx_out_ptr_{}".format(self.global_id), "x_list_ptr_{}".format(self.global_id), "firstx_width_list_ptr_{}".format(self.global_id), self.input_length_name])
            self.gpu_kernel_arg_list_next.extend([self.input_ptr_name+"_next", "firstx_out_ptr_{}_next".format(self.global_id), "x_list_ptr_{}".format(self.global_id), "firstx_width_list_ptr_{}".format(self.global_id), self.input_length_name+"_next"])

        elif self.op_name == "bucketize":
            # line 1: launch firstx kernel
            if_float = "1" if self.data_size == "args.batch_size" else "0" # dense -> float-firstx, sparse -> int64-firstx
            line_1 = "cuda_preprocess.bucketize_gpu_part({}, bucketize_out_ptr_{}, bucketize_border_list_ptr_{}, bucketize_length_list_ptr_{}, {}, {})".format(
                self.input_ptr_name, 
                self.global_id, 
                self.global_id, 
                self.global_id, 
                self.input_length_name, 
                self.data_size
            )
            line_1_next = "cuda_preprocess.bucketize_gpu_part({}_next, bucketize_out_ptr_{}_next, bucketize_border_list_ptr_{}, bucketize_length_list_ptr_{}, {}_next, {})".format(
                self.input_ptr_name, 
                self.global_id, 
                self.global_id, 
                self.global_id, 
                self.input_length_name, 
                self.data_size
            )
            self.gpu_part_code.append(line_1)
            self.gpu_part_code_next.append(line_1_next)
            self.gpu_kernel_arg_list.extend([self.input_ptr_name, "bucketize_out_ptr_{}".format(self.global_id), "bucketize_border_list_ptr_{}".format(self.global_id), "bucketize_length_list_ptr_{}".format(self.global_id), self.input_length_name])
            self.gpu_kernel_arg_list_next.extend([self.input_ptr_name+"_next", "bucketize_out_ptr_{}_next".format(self.global_id), "bucketize_border_list_ptr_{}".format(self.global_id), "bucketize_length_list_ptr_{}".format(self.global_id), self.input_length_name+"_next"])

        elif self.op_name == "boxcox":
            # line 1: launch boxcox kernel
            line_1 = "cuda_preprocess.boxcox_gpu_part({}, lambda_list_ptr_{}, {}, {})".format(
                self.input_ptr_name, 
                self.global_id, 
                self.input_length_name, 
                self.data_size
            )
            self.gpu_part_code.append(line_1)
        
        elif self.op_name == "clamp":
            # line 1: launch clamp kernel
            line_1 = "cuda_preprocess.clamp_gpu_part({}, clamp_min_list_ptr_{}, clamp_max_list_ptr_{}, {}, {})".format(
                self.input_ptr_name, 
                self.global_id, 
                self.global_id, 
                self.input_length_name, 
                self.data_size
            )
            self.gpu_part_code.append(line_1)
        
        elif self.op_name == "ngram":
            # line 1: launch ngram kernel
            line_1 = "cuda_preprocess.ngram_gpu_part({}, ngram_out_ptr_{}, ngram_width_ptr_{}, ngram_n_list_ptr_{}, {}, {})".format(
                self.input_ptr_name, 
                self.global_id, 
                self.global_id, 
                self.global_id, 
                self.input_length_name, 
                self.data_size
            )
            self.gpu_part_code.append(line_1)
        
        elif self.op_name == "onehot":
            # line 1: launch onehot kernel
            line_1 = "cuda_preprocess.onehot_gpu_part({}, onehot_out_ptr_{}, onehot_low_list_ptr_{}, onehot_high_list_ptr_{}, onehot_num_classes_list_ptr_{}, {}, {})".format(
                self.input_ptr_name, 
                self.global_id, 
                self.global_id, 
                self.global_id, 
                self.global_id,
                self.input_length_name,  
                self.data_size
            )
            self.gpu_part_code.append(line_1)

    def add_list(self):
        self.input_tensor_list_name_list = []
        self.output_tensor_list_name_list = []

class fused_kernel_decoder: # decoding the fused_kernel_list for latency prediction
    def __init__(self, nDev, fused_kernel_list, parse_graph, plan=0):
        # self.op_list = ["fill_null", "sigrid_hash", "bucketize", "logit", "firstx", "boxcox", "clamp", "onehot", "ngram", "mapid"]
        self.decoded_kernel_on_GPUs = [[] for _ in range(nDev)]
        self.nDev = nDev
        self.rate = 1
        if plan == 2:
            self.rate = 2
        elif plan == 3:
            self.rate = 4

        for i in range(nDev):
            this_kernel_list = fused_kernel_list[i]
            for this_kernel in this_kernel_list:
                self.decoded_kernel_on_GPUs[i].append(fused_kernel_node(this_kernel, parse_graph.nodes))
        
    def get_kernel_on_GPUs(self):
        return self.decoded_kernel_on_GPUs

    def print_fused_kernel_info(self):
        for gpu_id, i in enumerate(self.decoded_kernel_on_GPUs):
            print("GPU-{}:".format(gpu_id))
            print([(j.op_name, j.nOp) for j in i])

    def save_fused_kernel(self):
        with open("searched_fused_kernels/plan-{}_nGPU-{}.pkl".format(self.plan, self.nDev), "wb") as f:
            pickle.dump(self.decoded_kernel_on_GPUs, f)
        
    def duplicate_kernels(self):
        for gpu_id in range(self.nDev):
            self.decoded_kernel_on_GPUs[gpu_id] = self.decoded_kernel_on_GPUs[gpu_id] * self.rate
