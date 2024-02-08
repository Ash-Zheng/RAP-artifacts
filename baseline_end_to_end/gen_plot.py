import os
import re
import matplotlib.pyplot as plt


dir_prefix = "/workspace/RAP/baseline_end_to_end/"

def load_result(exp, nGPU, batch_size, plan):
    pattern = re.compile(r"total_throughput:(\d+\.\d+)")

    if exp == "CPU":
        path = os.path.join(dir_prefix, "CPU_based_baseline/result")
        file_name = "result_GPU-{}_Plan-{}_Batch-{}".format(nGPU, plan, batch_size)
    elif exp == "CUDA_stream":
        path = os.path.join(dir_prefix, "CUDA_stream/result")
        file_name = "result_GPU-{}_Plan-{}_Batch-{}".format(nGPU, plan, batch_size)
    else:
        path = os.path.join(dir_prefix, "{}/result".format(exp))
        file_name = "result_GPU-{}_Plan-{}_Batch-{}".format(nGPU, plan, batch_size)

    if_found = False
    for file in os.listdir(path):
        if file.startswith(file_name):
            if_found = True
            with open(os.path.join(path, file), "r") as f:
                content = f.read()
                match = pattern.search(content)
                if match:
                    # Extract and return the total_throughput value
                    return float(match.group(1))
                else:
                    print("total_throughput not found in the file.")
                    return None
    
    if if_found == False:
        print("Result file not found")
        return None


for batch_size in [4096, 8192]:
    for nGPU in [2, 4]: # machine for artifacts evaluation only have 4 GPUs
        for plan in [0, 1]:
            result_dict = {
                "CPU": 0.0,
                "CUDA_stream": 0.0,
                "MPS": 0.0,
                "RAP": 0.0
            }
            
            for key in result_dict.keys():
                num = load_result(key, nGPU, batch_size, plan)
                result_dict[key] = num
            
            # plot
            labels = list(result_dict.keys())
            values = list(result_dict.values())
            colors = ['red', 'green', 'blue', 'orange']

            plt.figure(figsize=(10, 6))  
            bars = plt.bar(labels, values, color=colors)  

            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

            plt.xlabel('Method')  
            plt.ylabel('Throughput') 
            plt.xticks(rotation=45)  

            plt.tight_layout()  
            plt.savefig("breakdown_batch_size-{}_plan-{}_nGPU-{}.png".format(batch_size, plan, nGPU), dpi=300)


