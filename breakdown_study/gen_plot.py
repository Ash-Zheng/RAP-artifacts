import os
import re
import matplotlib.pyplot as plt


dir_prefix = "/workspace/RAP/breakdown_study/"

def load_result(exp, batch_size, plan):
    pattern = re.compile(r"total_throughput:(\d+\.\d+)")

    if exp == "MPS":
        path = os.path.join(dir_prefix, "MPS_and_sequential/result")
        file_name = "MPS-result_GPU-4_Plan-{}_Batch-{}".format(plan, batch_size)
    elif exp == "sequential":
        path = os.path.join(dir_prefix, "MPS_and_sequential/result")
        file_name = "Sequential-result_GPU-4_Plan-{}_Batch-{}".format(plan, batch_size)
    else:
        path = os.path.join(dir_prefix, "{}/result".format(exp))
        file_name = "result_GPU-4_Plan-{}_Batch-{}".format(plan, batch_size)

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
                    return 0.1
    
    if if_found == False:
        print("Result file not found")
        return 0.1


nGPU = 4
for batch_size in [4096, 8192]:
    for plan in [0, 1]:
        result_dict = {
            "sequential": 0.0,
            "ideal": 0.0,
            "MPS": 0.0,
            "no_fusion": 0.0,
            "no_mapping": 0.0,
            "RAP": 0.0,
        }
        
        for key in result_dict.keys():
            num = load_result(key, batch_size, plan)
            result_dict[key] = num
        
        # normalize result:
        base = result_dict["sequential"]
        for key in result_dict.keys():
            result_dict[key] = result_dict[key] / base
        
        # plot
        labels = list(result_dict.keys())
        values = list(result_dict.values())
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']

        plt.figure(figsize=(10, 6))  
        bars = plt.bar(labels, values, color=colors)  

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

        plt.xlabel('Method')  
        plt.ylabel('Normalized Speedup') 
        plt.xticks(rotation=45)  

        plt.tight_layout()  
        plt.savefig("breakdown_batch_size-{}_plan-{}.png".format(batch_size, plan), dpi=300)


