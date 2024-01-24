import os
import shutil
import jinja2

class code_generator:
    def __init__(self, nDev, plan_id, dlrm_layers, folder_name="generated_code"):
        self.nDev = nDev
        self.plan_id = plan_id
        self.dlrm_layers = dlrm_layers
        self.dense_output_code = [["dense_input_list = ["] for _ in range(self.nDev)]
        self.sparse_output_code = [["sparse_input_list = ["] for _ in range(self.nDev)]
        self.dense_output_code_next = [["dense_input_list_next = ["] for _ in range(self.nDev)]
        self.sparse_output_code_next = [["sparse_input_list_next = ["] for _ in range(self.nDev)]
        self.sparse_comm_code = [[] for _ in range(self.nDev)]
        self.folder_name = folder_name

        # first iter code
        self.first_iter_code = ["" for _ in range(self.nDev)]

        # input list and cpu code (thread function code)
        self.input_cpu_code = ["" for _ in range(self.nDev)]

        # input var list and output var list for thread function
        self.input_var_list = [[] for _ in range(self.nDev)]
        self.output_var_list = [[] for _ in range(self.nDev)]

        # input&output encode and decode code for thread function
        self.input_encode_code = ["" for _ in range(self.nDev)]
        self.input_decode_code = ["" for _ in range(self.nDev)]
        self.output_encode_code = ["" for _ in range(self.nDev)]
        self.output_decode_code = ["" for _ in range(self.nDev)]

        # gpu_part_code 
        self.gpu_code = ["" for _ in range(self.nDev)]

        # output_tensor var list and code
        self.dense_output_tensor_var_list = [[] for _ in range(self.nDev)]
        self.dense_output_tensor_code = ["" for _ in range(self.nDev)]
        self.sparse_output_tensor_var_list = [[] for _ in range(self.nDev)]
        self.sparse_output_tensor_code = ["" for _ in range(self.nDev)]

        # msg code
        self.msg_code = ["" for _ in range(len(self.dlrm_layers))]

    def remove_files_in_folder(self, folder):
        # Check if the folder exists
        if not os.path.exists(folder):
            print(f"The folder '{folder}' does not exist.")
            return

        # Loop through each item in the folder
        for item_name in os.listdir(folder):
            item_path = os.path.join(folder, item_name)

            # Check if the item is a file and delete it
            if os.path.isfile(item_path):
                os.remove(item_path)
                print(f"File {item_path} has been removed.")
            
            # If the item is a directory, remove the directory and its contents
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Directory {item_path} has been removed.")


    def init_msg_code(self, schedule):
        for gpu_id in range(self.nDev):
            for layer_id in range(len(self.dlrm_layers)):
                if len(schedule[gpu_id][layer_id]) > 0:
                    self.msg_code[layer_id] = "sharded_emts.signal_layer.register_full_backward_hook(queue_hook_func) # register init queue_hook_func" if layer_id == 3 else "queue_list[rank].put(1)"
                    


    # generate code for kernels, the code is stored in the kernel object
    def init_code_for_GPUs(self, schedule, all_node):
        global_kernel_id = 0
        for gpu_id in range(self.nDev):
            for layer_id in range(len(self.dlrm_layers)):
                for kernel_id, this_kernel in enumerate(schedule[gpu_id][layer_id]):
                    if this_kernel.global_id != -1:
                        this_kernel.add_list()
                        # continue
                    # updata fused information in the parsed preprocessing graph
                    for idx, node_id in enumerate(this_kernel.node_list):
                        this_node = all_node[node_id]
                        this_node.set_fused_place(gpu_id, layer_id, kernel_id, idx)

                    if this_kernel.op_name in [
                        "fill_null_dense", "fill_null_sparse", 
                        "logit", "sigrid_hash", "firstx", "bucketize", "boxcox", "clamp", "mapid", "ngram", "onehot"
                    ]:
                        this_kernel.global_id = global_kernel_id
                        this_kernel.init_input_name(all_node, schedule)
                        this_kernel.init_input_list_code(all_node, schedule)
                        this_kernel.init_data_prepare_code(all_node)
                        this_kernel.init_cpu_part_code(all_node, schedule)
                        this_kernel.init_gpu_part_code()
                    
                    # for idx, node_id in enumerate(this_kernel.node_list):
                    #     this_node = all_node[node_id]
                    #     if len(this_node.successor) == 0: # output node
                    #         self.add_output_code(gpu_id, idx, this_kernel)

                    global_kernel_id += 1
                    

    def append_code(self, output_code_list, idx,  kernel_code, indent=1):
        for line in kernel_code:
            output_code_list[idx] += "    " * indent + line
            output_code_list[idx] += "\n"
    

    def get_code_first_iter(self, schedule): # full data preprocessing of the first batch
        for gpu_id in range(self.nDev):
            for layer_id in range(len(self.dlrm_layers)):
                for _, this_kernel in enumerate(schedule[gpu_id][layer_id]):
                    self.append_code(self.first_iter_code, gpu_id, ["# {}-{} input_list_code:".format(this_kernel.op_name, this_kernel.global_id)], indent=3)
                    self.append_code(self.first_iter_code, gpu_id, this_kernel.input_list_code, indent=3)
                    self.append_code(self.first_iter_code, gpu_id, ["# {}-{} data_prepare_code:".format(this_kernel.op_name, this_kernel.global_id)], indent=3)
                    self.append_code(self.first_iter_code, gpu_id, this_kernel.data_prepare_code, indent=3)
                    self.append_code(self.first_iter_code, gpu_id, ["# {}-{} cpu_part_code:".format(this_kernel.op_name, this_kernel.global_id)], indent=3)
                    self.append_code(self.first_iter_code, gpu_id, this_kernel.cpu_part_code, indent=3)

                    if this_kernel.op_name == "fill_null_dense" or this_kernel.op_name == "fill_null_sparse":
                        # self.input_var_list[gpu_id].append(this_kernel.output_tensor_list_name)
                        # self.output_var_list[gpu_id].append(this_kernel.input_tensor_list_name + "_tensor")

                        for name in this_kernel.output_tensor_list_name_list:
                            self.input_var_list[gpu_id].append(name)
                        for name in this_kernel.input_tensor_list_name_list:
                            self.output_var_list[gpu_id].append(name + "_tensor")
                    elif this_kernel.op_name == "logit":
                        self.output_var_list[gpu_id].append(this_kernel.input_ptr_name)
                        self.output_var_list[gpu_id].append(this_kernel.input_length_name)
                    elif this_kernel.op_name == "sigrid_hash":
                        offset_name = this_kernel.op_name + "_offset_length_{}".format(this_kernel.global_id)
                        self.output_var_list[gpu_id].append(this_kernel.input_ptr_name)
                        self.output_var_list[gpu_id].append(offset_name)
                        self.output_var_list[gpu_id].append(this_kernel.input_length_name)
                    elif this_kernel.op_name == "firstx":
                        self.input_var_list[gpu_id].append("x_list_{}".format(this_kernel.global_id))
                        out_ptr_name = "firstx_out_ptr_{}".format(this_kernel.global_id)
                        self.output_var_list[gpu_id].append(this_kernel.input_ptr_name) 
                        self.output_var_list[gpu_id].append(out_ptr_name)
                        self.output_var_list[gpu_id].append(this_kernel.input_length_name)
                        # self.output_var_list[gpu_id].append(this_kernel.output_tensor_list_name)
                        for name in this_kernel.output_tensor_list_name_list:
                            self.output_var_list[gpu_id].append(name)
                    elif this_kernel.op_name == "bucketize":
                        self.input_var_list[gpu_id].append("border_list_{}".format(this_kernel.global_id))
                        out_ptr_name = "bucketize_out_ptr_{}".format(this_kernel.global_id)
                        self.output_var_list[gpu_id].append(this_kernel.input_ptr_name) 
                        self.output_var_list[gpu_id].append(out_ptr_name) 
                        self.output_var_list[gpu_id].append(this_kernel.input_length_name)
                        # self.output_var_list[gpu_id].append(this_kernel.output_tensor_list_name)
                        for name in this_kernel.output_tensor_list_name_list:
                            self.output_var_list[gpu_id].append(name)
                    elif this_kernel.op_name == "boxcox":
                        self.output_var_list[gpu_id].append(this_kernel.input_ptr_name)
                        self.output_var_list[gpu_id].append(this_kernel.input_length_name)
                    elif this_kernel.op_name == "clamp":
                        self.output_var_list[gpu_id].append(this_kernel.input_ptr_name)
                        self.output_var_list[gpu_id].append(this_kernel.input_length_name)
                    elif this_kernel.op_name == "ngram":
                        self.input_var_list[gpu_id].append("ngram_n_list_{}".format(this_kernel.global_id))
                        out_ptr_name = "ngram_out_ptr_{}".format(this_kernel.global_id)
                        self.output_var_list[gpu_id].append(this_kernel.input_ptr_name)
                        self.output_var_list[gpu_id].append(out_ptr_name)
                        self.output_var_list[gpu_id].append(this_kernel.input_length_name)
                        # self.output_var_list[gpu_id].append(this_kernel.output_tensor_list_name)
                        for name in this_kernel.output_tensor_list_name_list:
                            self.output_var_list[gpu_id].append(name)
                    elif this_kernel.op_name == "onehot":
                        self.input_var_list[gpu_id].append("onehot_num_classes_list_{}".format(this_kernel.global_id))
                        out_ptr_name = "onehot_out_ptr_{}".format(this_kernel.global_id)
                        self.output_var_list[gpu_id].append(this_kernel.input_ptr_name)
                        self.output_var_list[gpu_id].append(out_ptr_name)
                        self.output_var_list[gpu_id].append(this_kernel.input_length_name)
                        # self.output_var_list[gpu_id].append(this_kernel.output_tensor_list_name)
                        for name in this_kernel.output_tensor_list_name_list:
                            self.output_var_list[gpu_id].append(name)


            self.append_code(self.first_iter_code, gpu_id, ["# =========================================================================="], indent=3)
            self.append_code(self.first_iter_code, gpu_id, ["# =========================== First Iter finished =========================="], indent=3)
            self.append_code(self.first_iter_code, gpu_id, ["# =========================================================================="], indent=3)
        
        for gpu_id in range(self.nDev): # input encode and decode
            this_code = "input_list = ["
            decode_code = ""
            for var_name in self.input_var_list[gpu_id]:
                this_code += var_name + ", "
                decode_code += var_name + ", "
            this_code = this_code[:-2] + "]"
            decode_code = decode_code[:-2] + " = input_list"
            self.append_code(self.input_encode_code, gpu_id, [this_code], indent=3)
            self.append_code(self.input_decode_code, gpu_id, [decode_code], indent=1)
        
        for gpu_id in range(self.nDev): # output encode and decode
            this_code = "output_list.extend(["
            decode_code = ""
            for var_name in self.output_var_list[gpu_id]:
                this_code += var_name + ", "
                decode_code += var_name + ", "
            this_code = this_code[:-2] + "])"
            decode_code = decode_code[:-2] + " = output_list"
            self.append_code(self.output_encode_code, gpu_id, [this_code], indent=1)

            # return num_check:
            check_str = "if len(output_list) == {}:".format(len(self.output_var_list[gpu_id]))
            self.append_code(self.output_decode_code, gpu_id, [check_str], indent=4)
            self.append_code(self.output_decode_code, gpu_id, [decode_code], indent=5)


    def get_code_for_input_cpu(self, schedule): # input and cpu part code in the thread function
        indent = 2
        for gpu_id in range(self.nDev):
            for layer_id in range(len(self.dlrm_layers)):
                for _, this_kernel in enumerate(schedule[gpu_id][layer_id]):
                    self.append_code(self.input_cpu_code, gpu_id, ["# {}-{} input_list_code:".format(this_kernel.op_name, this_kernel.global_id)], indent=indent)
                    self.append_code(self.input_cpu_code, gpu_id, this_kernel.input_list_code, indent=indent)
                    self.append_code(self.input_cpu_code, gpu_id, ["# {}-{} cpu_part_code:".format(this_kernel.op_name, this_kernel.global_id)], indent=indent)
                    self.append_code(self.input_cpu_code, gpu_id, this_kernel.cpu_part_code, indent=indent)


    def get_code_for_gpu(self, schedule): # gpu part code
        indent = 4
        for gpu_id in range(self.nDev):
            for layer_id in range(len(self.dlrm_layers)):
                if len(schedule[gpu_id][layer_id]) > 0:
                    self.append_code(self.gpu_code, gpu_id, ["if i > 0:"], indent=indent)
                    self.append_code(self.gpu_code, gpu_id, ["queue_wait(queue_list[this_rank])"], indent=indent+1)
                    self.append_code(self.gpu_code, gpu_id, ["for _ in range(dup_times):"], indent=indent)

                for _, this_kernel in enumerate(schedule[gpu_id][layer_id]):
                    self.append_code(self.gpu_code, gpu_id, ["# {}-{} gpu_code:".format(this_kernel.op_name, this_kernel.global_id)], indent=indent+1)
                    self.append_code(self.gpu_code, gpu_id, this_kernel.gpu_part_code, indent=indent+1)


    # def add_output_code(self, gpu_id, node_idx, kernel):
    #     if kernel.data_size == "args.batch_size": # dense out
    #         code = self.dense_output_code[gpu_id][0]
    #         code_next = self.dense_output_code_next[gpu_id][0]
    #     else:
    #         code = self.sparse_output_code[gpu_id][0]
    #         code_next = self.sparse_output_code_next[gpu_id][0]
        
    #     code += "{}[{}], ".format(kernel.output_tensor_list_name, node_idx)
    #     code_next += "{}[{}], ".format(kernel.output_tensor_list_name + "_next", node_idx)

    #     if kernel.data_size == "args.batch_size": # dense out
    #         self.dense_output_code[gpu_id][0] = code
    #         self.dense_output_code_next[gpu_id][0] = code_next
    #     else:
    #         self.sparse_output_code[gpu_id][0] = code
    #         self.sparse_output_code_next[gpu_id][0] = code_next


    def get_code_for_output_tensor(self, schedule, all_node):
        for gpu_id in range(self.nDev):
            for layer_id in range(len(self.dlrm_layers)):
                for kernel_id, this_kernel in enumerate(schedule[gpu_id][layer_id]):
                    for idx, node_id in enumerate(this_kernel.node_list):
                        this_node = all_node[node_id]
                        if len(this_node.successor) == 0: # output node
                            var_name = "{}[{}]".format(this_kernel.output_tensor_list_name, idx)
                            if this_kernel.data_size == "args.batch_size": # dense out
                                self.dense_output_tensor_var_list[gpu_id].append(var_name)
                            else:
                                self.sparse_output_tensor_var_list[gpu_id].append(var_name)
        
        for gpu_id in range(self.nDev):
            dense_out_code = "dense_input_list = ["
            sparse_out_code = "sparse_input_list = ["
            for var_name in self.dense_output_tensor_var_list[gpu_id]:
                dense_out_code += var_name + ", "
            for var_name in self.sparse_output_tensor_var_list[gpu_id]:
                sparse_out_code += var_name + ", "
            dense_out_code = dense_out_code[:-2] + "]"
            sparse_out_code = sparse_out_code[:-2] + "]"
            self.append_code(self.dense_output_tensor_code, gpu_id, [dense_out_code], indent=4)
            self.append_code(self.sparse_output_tensor_code, gpu_id, [sparse_out_code], indent=4)


    def init_all_codes(self, schedule, all_node):
        self.init_msg_code(schedule)
        self.init_code_for_GPUs(schedule, all_node)
        self.get_code_first_iter(schedule)
        self.get_code_for_input_cpu(schedule)
        self.get_code_for_gpu(schedule)
        self.get_code_for_output_tensor(schedule, all_node)


    def RAP_gen_code_all(self): # generate code for RAP
        templateLoader = jinja2.FileSystemLoader(searchpath="./")
        templateEnv = jinja2.Environment(loader=templateLoader)

        TEMPLATE_FILE = "template/GPU_{}_template_all.jinja".format(self.nDev)
        template = templateEnv.get_template(TEMPLATE_FILE)

        if self.nDev == 2:
            templateVars = {
                # ========= first iter =========
                "gpu_0_first_iter": self.first_iter_code[0],
                "gpu_1_first_iter": self.first_iter_code[1],

                # ========= input&output encode decode =========
                "gpu_0_input_list_encode": self.input_encode_code[0],
                "gpu_1_input_list_encode": self.input_encode_code[1],
                "gpu_0_input_list_decode": self.input_decode_code[0],
                "gpu_1_input_list_decode": self.input_decode_code[1],

                "gpu_0_output_list_encode": self.output_encode_code[0],
                "gpu_1_output_list_encode": self.output_encode_code[1],
                "gpu_0_output_list_decode": self.output_decode_code[0],
                "gpu_1_output_list_decode": self.output_decode_code[1],

                # ========= thread function =========
                "gpu_0_input_cpu_code": self.input_cpu_code[0],
                "gpu_1_input_cpu_code": self.input_cpu_code[1],

                # ========= gpu code =========
                "gpu_0_gpu_code": self.gpu_code[0],
                "gpu_1_gpu_code": self.gpu_code[1],

                # ========= output tensor =========
                "gpu_0_dense_output_tensor": self.dense_output_tensor_code[0],
                "gpu_1_dense_output_tensor": self.dense_output_tensor_code[1],
                "gpu_0_sparse_output_tensor": self.sparse_output_tensor_code[0],
                "gpu_1_sparse_output_tensor": self.sparse_output_tensor_code[1],

                # ========= msg code =========
                "put_code_0": self.msg_code[0],
                "put_code_1": self.msg_code[1],
                "put_code_2": self.msg_code[2],
                "put_code_3": self.msg_code[3],
                "put_code_4": self.msg_code[4],
            }
        elif self.nDev == 4:
            templateVars = {
                # ========= first iter =========
                "gpu_0_first_iter": self.first_iter_code[0],
                "gpu_1_first_iter": self.first_iter_code[1],
                "gpu_2_first_iter": self.first_iter_code[2],
                "gpu_3_first_iter": self.first_iter_code[3],

                # ========= input&output encode decode =========
                "gpu_0_input_list_encode": self.input_encode_code[0],
                "gpu_1_input_list_encode": self.input_encode_code[1],
                "gpu_2_input_list_encode": self.input_encode_code[2],
                "gpu_3_input_list_encode": self.input_encode_code[3],
                "gpu_0_input_list_decode": self.input_decode_code[0],
                "gpu_1_input_list_decode": self.input_decode_code[1],
                "gpu_2_input_list_decode": self.input_decode_code[2],
                "gpu_3_input_list_decode": self.input_decode_code[3],

                "gpu_0_output_list_encode": self.output_encode_code[0],
                "gpu_1_output_list_encode": self.output_encode_code[1],
                "gpu_2_output_list_encode": self.output_encode_code[2],
                "gpu_3_output_list_encode": self.output_encode_code[3],
                "gpu_0_output_list_decode": self.output_decode_code[0],
                "gpu_1_output_list_decode": self.output_decode_code[1],
                "gpu_2_output_list_decode": self.output_decode_code[2],
                "gpu_3_output_list_decode": self.output_decode_code[3],

                # ========= thread function =========
                "gpu_0_input_cpu_code": self.input_cpu_code[0],
                "gpu_1_input_cpu_code": self.input_cpu_code[1],
                "gpu_2_input_cpu_code": self.input_cpu_code[2],
                "gpu_3_input_cpu_code": self.input_cpu_code[3],

                # ========= gpu code =========
                "gpu_0_gpu_code": self.gpu_code[0],
                "gpu_1_gpu_code": self.gpu_code[1],
                "gpu_2_gpu_code": self.gpu_code[2],
                "gpu_3_gpu_code": self.gpu_code[3],

                # ========= output tensor =========
                "gpu_0_dense_output_tensor": self.dense_output_tensor_code[0],
                "gpu_1_dense_output_tensor": self.dense_output_tensor_code[1],
                "gpu_2_dense_output_tensor": self.dense_output_tensor_code[2],
                "gpu_3_dense_output_tensor": self.dense_output_tensor_code[3],
                "gpu_0_sparse_output_tensor": self.sparse_output_tensor_code[0],
                "gpu_1_sparse_output_tensor": self.sparse_output_tensor_code[1],
                "gpu_2_sparse_output_tensor": self.sparse_output_tensor_code[2],
                "gpu_3_sparse_output_tensor": self.sparse_output_tensor_code[3],

                # ========= msg code =========
                "put_code_0": self.msg_code[0],
                "put_code_1": self.msg_code[1],
                "put_code_2": self.msg_code[2],
                "put_code_3": self.msg_code[3],
                "put_code_4": self.msg_code[4],
            }
        elif self.nDev == 8:
            templateVars = {
                # ========= first iter =========
                "gpu_0_first_iter": self.first_iter_code[0],
                "gpu_1_first_iter": self.first_iter_code[1],
                "gpu_2_first_iter": self.first_iter_code[2],
                "gpu_3_first_iter": self.first_iter_code[3],
                "gpu_4_first_iter": self.first_iter_code[4],
                "gpu_5_first_iter": self.first_iter_code[5],
                "gpu_6_first_iter": self.first_iter_code[6],
                "gpu_7_first_iter": self.first_iter_code[7],

                # ========= input&output encode decode =========
                "gpu_0_input_list_encode": self.input_encode_code[0],
                "gpu_1_input_list_encode": self.input_encode_code[1],
                "gpu_2_input_list_encode": self.input_encode_code[2],
                "gpu_3_input_list_encode": self.input_encode_code[3],
                "gpu_4_input_list_encode": self.input_encode_code[4],
                "gpu_5_input_list_encode": self.input_encode_code[5],
                "gpu_6_input_list_encode": self.input_encode_code[6],
                "gpu_7_input_list_encode": self.input_encode_code[7],
                "gpu_0_input_list_decode": self.input_decode_code[0],
                "gpu_1_input_list_decode": self.input_decode_code[1],
                "gpu_2_input_list_decode": self.input_decode_code[2],
                "gpu_3_input_list_decode": self.input_decode_code[3],
                "gpu_4_input_list_decode": self.input_decode_code[4],
                "gpu_5_input_list_decode": self.input_decode_code[5],
                "gpu_6_input_list_decode": self.input_decode_code[6],
                "gpu_7_input_list_decode": self.input_decode_code[7],

                "gpu_0_output_list_encode": self.output_encode_code[0],
                "gpu_1_output_list_encode": self.output_encode_code[1],
                "gpu_2_output_list_encode": self.output_encode_code[2],
                "gpu_3_output_list_encode": self.output_encode_code[3],
                "gpu_4_output_list_encode": self.output_encode_code[4],
                "gpu_5_output_list_encode": self.output_encode_code[5],
                "gpu_6_output_list_encode": self.output_encode_code[6],
                "gpu_7_output_list_encode": self.output_encode_code[7],
                "gpu_0_output_list_decode": self.output_decode_code[0],
                "gpu_1_output_list_decode": self.output_decode_code[1],
                "gpu_2_output_list_decode": self.output_decode_code[2],
                "gpu_3_output_list_decode": self.output_decode_code[3],
                "gpu_4_output_list_decode": self.output_decode_code[4],
                "gpu_5_output_list_decode": self.output_decode_code[5],
                "gpu_6_output_list_decode": self.output_decode_code[6],
                "gpu_7_output_list_decode": self.output_decode_code[7],

                # ========= thread function =========
                "gpu_0_input_cpu_code": self.input_cpu_code[0],
                "gpu_1_input_cpu_code": self.input_cpu_code[1],
                "gpu_2_input_cpu_code": self.input_cpu_code[2],
                "gpu_3_input_cpu_code": self.input_cpu_code[3],
                "gpu_4_input_cpu_code": self.input_cpu_code[4],
                "gpu_5_input_cpu_code": self.input_cpu_code[5],
                "gpu_6_input_cpu_code": self.input_cpu_code[6],
                "gpu_7_input_cpu_code": self.input_cpu_code[7],

                # ========= gpu code =========
                "gpu_0_gpu_code": self.gpu_code[0],
                "gpu_1_gpu_code": self.gpu_code[1],
                "gpu_2_gpu_code": self.gpu_code[2],
                "gpu_3_gpu_code": self.gpu_code[3],
                "gpu_4_gpu_code": self.gpu_code[4],
                "gpu_5_gpu_code": self.gpu_code[5],
                "gpu_6_gpu_code": self.gpu_code[6],
                "gpu_7_gpu_code": self.gpu_code[7],

                # ========= output tensor =========
                "gpu_0_dense_output_tensor": self.dense_output_tensor_code[0],
                "gpu_1_dense_output_tensor": self.dense_output_tensor_code[1],
                "gpu_2_dense_output_tensor": self.dense_output_tensor_code[2],
                "gpu_3_dense_output_tensor": self.dense_output_tensor_code[3],
                "gpu_4_dense_output_tensor": self.dense_output_tensor_code[4],
                "gpu_5_dense_output_tensor": self.dense_output_tensor_code[5],
                "gpu_6_dense_output_tensor": self.dense_output_tensor_code[6],
                "gpu_7_dense_output_tensor": self.dense_output_tensor_code[7],
                "gpu_0_sparse_output_tensor": self.sparse_output_tensor_code[0],
                "gpu_1_sparse_output_tensor": self.sparse_output_tensor_code[1],
                "gpu_2_sparse_output_tensor": self.sparse_output_tensor_code[2],
                "gpu_3_sparse_output_tensor": self.sparse_output_tensor_code[3],
                "gpu_4_sparse_output_tensor": self.sparse_output_tensor_code[4],
                "gpu_5_sparse_output_tensor": self.sparse_output_tensor_code[5],
                "gpu_6_sparse_output_tensor": self.sparse_output_tensor_code[6],
                "gpu_7_sparse_output_tensor": self.sparse_output_tensor_code[7],

                # ========= msg code =========
                "put_code_0": self.msg_code[0],
                "put_code_1": self.msg_code[1],
                "put_code_2": self.msg_code[2],
                "put_code_3": self.msg_code[3],
                "put_code_4": self.msg_code[4],
            }

        outputText = template.render( templateVars )

        output_file = "combined_code/GPU_{}_plan_{}_all.py".format(self.nDev, self.plan_id)
        with open(output_file, "w") as f:
            f.write(outputText)