import os
import sys
sys.path.append('/workspace/RAP/torchrec_models')

import xgboost as xgb
import numpy as np
import torch
from dlrm_parser import get_args
from scipy.interpolate import griddata
import pickle

def train_data_prepare_predictor(args, if_save):
    all_op_list =  ["fill_null", "sigrid_hash", "bucketize", "logit", "firstx", "boxcox", "clamp", "onehot", "ngram", "mapid"]

    train_dict = {
        "1d": ["sigrid_hash", "logit", "boxcox", "clamp", "mapid"],
        "fill_null": ["fill_null"],
        "bucketize": ["bucketize"],
        "firstx": ["firstx"],
        "onehot": ["onehot"],
        "ngram": ["ngram"],
    }

    accuracy_dict = {}
    total_n_data = 0

    for key in train_dict.keys():
        op_list = train_dict[key]

        # ================================================================================
        # ============================ Original Data select ==============================
        # ================================================================================
        all_data = []
        all_result = []
        all_data_test = []
        all_result_test = []
        for op in op_list:
            file_name = "kernel_latency/{}/{}_latency.pt".format(args.batch_size, op)
            test_file_name = "kernel_latency/{}/{}_latency_test.pt".format(args.batch_size, op)
            
            data_and_result = torch.load(file_name)
            data_and_result_test = torch.load(test_file_name)

            concated_data = torch.cat((data_and_result, data_and_result_test))
            indices = torch.randperm(concated_data.size(0))
            shuffled_data_and_result = concated_data[indices]

            total_n_data += concated_data.size(0)

            #split train and test set
            nTrain = int(0.9 * concated_data.size(0))
            train_data = shuffled_data_and_result[0:nTrain]
            test_data = shuffled_data_and_result[nTrain:]
            op_idx = all_op_list.index(op)

            data = train_data[:, 0:2]
            result = train_data[:, 2]
            op_idx_tensor = torch.zeros((data.shape[0], 1)) + op_idx
            data = torch.cat((op_idx_tensor, data), 1)

            all_data.append(data)
            all_result.append(result)

            data = test_data[:, 0:2]
            result = test_data[:, 2]
            op_idx_tensor = torch.zeros((data.shape[0], 1)) + op_idx
            data = torch.cat((op_idx_tensor, data), 1)

            all_data_test.append(data)
            all_result_test.append(result)

        all_data = torch.cat(all_data, 0)
        all_result = torch.cat(all_result, 0)
        all_data_test = torch.cat(all_data_test, 0)
        all_result_test = torch.cat(all_result_test, 0)

        print("all_data:", all_data.shape)
        print("all_result:", all_result.shape)
        print("all_data_test:", all_data_test.shape)
        print("all_result_test:", all_result_test.shape)

        # Create a DMatrix
        all_data = np.array(all_data)
        all_result = np.array(all_result)
        all_data_test = np.array(all_data_test)
        all_result_test = np.array(all_result_test)
        dtrain = xgb.DMatrix(all_data, label=all_result)
        dtest = xgb.DMatrix(all_data_test, label=all_result_test)

        params = {
            'objective': 'reg:squarederror',
            'max_depth': 5,
            'learning_rate': 0.05,
            'silent': 1,
            'n_estimators': 300
        }

        bst = xgb.train(params, dtrain, num_boost_round=300, early_stopping_rounds=10, evals=[(dtest, 'test')])
        

        output_info = []
        test_cnt = 0
        error_cnt = 0
        y_pred = bst.predict(dtest)
        for i in range(len(y_pred)):
            test_cnt += 1
            if abs(all_result_test[i] - y_pred[i])/all_result_test[i] > 0.1:
                output_info.append("result:{:.4f}, y_pred:{:.4f}, diff:{:.4f}".format(all_result_test[i], y_pred[i], (all_result_test[i] - y_pred[i])/all_result_test[i]))
                error_cnt += 1
        
        

        train_cnt = 0
        train_error_cnt = 0
        y_pred = bst.predict(dtrain)
        for i in range(len(y_pred)):
            train_cnt += 1
            if abs(all_result[i] - y_pred[i])/all_result[i] > 0.1:
                output_info.append("result:{:.4f}, y_pred:{:.4f}, diff:{:.4f}".format(all_result[i], y_pred[i], (all_result[i] - y_pred[i])/all_result[i]))
                train_error_cnt += 1

        # for info in output_info:
        #     print(info)

        print("test_test: n_test:{}, n_correct:{}, accuracy:{:.3f}".format(test_cnt, test_cnt-error_cnt, (test_cnt-error_cnt)/test_cnt))
        print("train_test: n_test:{}, n_correct:{}, accuracy:{:.3f}".format(train_cnt, train_cnt-train_error_cnt, (train_cnt-train_error_cnt)/train_cnt))
        print("total: n_test:{}, n_correct:{}, accuracy:{:.3f}".format(train_cnt + test_cnt, train_cnt + test_cnt - error_cnt - train_error_cnt, (train_cnt + test_cnt - error_cnt - train_error_cnt)/(train_cnt + test_cnt)))
        accuracy_dict[key] = [round((test_cnt-error_cnt)/test_cnt, 3), round((train_cnt-train_error_cnt)/train_cnt, 3), round((train_cnt + test_cnt - error_cnt - train_error_cnt)/(train_cnt + test_cnt), 3)]
        if if_save:
            predict_model_file_name = "data_prepare_latency/{}_data_prepare_predictor_{}.bst".format(key, args.batch_size)
            bst.save_model(predict_model_file_name)

    acc_list = ["test_acc", "train_acc", "total_acc"]
    for key in accuracy_dict:
        for idx, acc in enumerate(accuracy_dict[key]):
            print("{}_{}: {}".format(key, acc_list[idx], acc))


def train_predicor(args, if_save):
    all_op_list =  ["fill_null", "sigrid_hash", "bucketize", "logit", "firstx", "boxcox", "clamp", "onehot", "ngram", "mapid"]

    train_dict = {
        "1d": ["fill_null", "sigrid_hash", "logit", "boxcox", "clamp", "mapid"],
        # "1d": ["sigrid_hash", "bucketize", "logit", "boxcox", "clamp", "mapid"],
        # "fill_null": ["fill_null"],
        "bucketize": ["bucketize"],
        "firstx": ["firstx"], 
        "ngram": ["ngram"], 
        "onehot": ["onehot"]
    }

    accuracy_dict = {}
    total_n_data = 0

    for key in train_dict.keys():
        op_list = train_dict[key]

        # ================================================================================
        # ============================ Original Data select ==============================
        # ================================================================================
        all_data = []
        all_result = []
        all_data_test = []
        all_result_test = []
        for op in op_list:
            file_name = "kernel_latency/{}/{}_latency.pt".format(args.batch_size, op)
            test_file_name = "kernel_latency/{}/{}_latency_test.pt".format(args.batch_size, op)
            
            data_and_result = torch.load(file_name)
            data_and_result_test = torch.load(test_file_name)

            concated_data = torch.cat((data_and_result, data_and_result_test))
            indices = torch.randperm(concated_data.size(0))
            shuffled_data_and_result = concated_data[indices]

            total_n_data += concated_data.size(0)

            #split train and test set
            nTrain = int(0.9 * concated_data.size(0))
            train_data = shuffled_data_and_result[0:nTrain]
            test_data = shuffled_data_and_result[nTrain:]
            op_idx = all_op_list.index(op)

            data = train_data[:, 0:2]
            result = train_data[:, 2]
            op_idx_tensor = torch.zeros((data.shape[0], 1)) + op_idx
            data = torch.cat((op_idx_tensor, data), 1)

            all_data.append(data)
            all_result.append(result)

            data = test_data[:, 0:2]
            result = test_data[:, 2]
            op_idx_tensor = torch.zeros((data.shape[0], 1)) + op_idx
            data = torch.cat((op_idx_tensor, data), 1)

            all_data_test.append(data)
            all_result_test.append(result)

        all_data = torch.cat(all_data, 0)
        all_result = torch.cat(all_result, 0)
        all_data_test = torch.cat(all_data_test, 0)
        all_result_test = torch.cat(all_result_test, 0)

        # ================================================================================
        # ============================ Original Data select ==============================
        # ================================================================================
        # all_data = []
        # all_result = []
        # # conbine training data
        # for op in op_list:
        #     file_name = "kernel_latency/{}/{}_latency.pt".format(args.batch_size, op)
        #     op_idx = all_op_list.index(op)

        #     data_and_result = torch.load(file_name)

        #     data = data_and_result[:, 0:2]
        #     result = data_and_result[:, 2]

        #     op_idx_tensor = torch.zeros((data.shape[0], 1)) + op_idx
        #     data = torch.cat((op_idx_tensor, data), 1)

        #     all_data.append(data)
        #     all_result.append(result)

        # all_data = torch.cat(all_data, 0)
        # all_result = torch.cat(all_result, 0)


        # all_data_test = []
        # all_result_test = []
        # # conbine test data
        # for op in op_list:
        #     file_name = "kernel_latency/{}/{}_latency_test.pt".format(args.batch_size, op)
        #     op_idx = all_op_list.index(op)

        #     data_and_result = torch.load(file_name)

        #     data = data_and_result[:, 0:2]
        #     result = data_and_result[:, 2]

        #     op_idx_tensor = torch.zeros((data.shape[0], 1)) + op_idx
        #     data = torch.cat((op_idx_tensor, data), 1)

        #     all_data_test.append(data)
        #     all_result_test.append(result)

        # all_data_test = torch.cat(all_data_test, 0)
        # all_result_test = torch.cat(all_result_test, 0)
        # ================================================================================
        # ============================ Original Data select ==============================
        # ================================================================================
        # all_data: [[fuse_length/nBatch], [op_width]]
        print("all_data:", all_data.shape)
        print("all_result:", all_result.shape)
        print("all_data_test:", all_data_test.shape)
        print("all_result_test:", all_result_test.shape)

        # Create a DMatrix
        all_data = np.array(all_data)
        all_result = np.array(all_result)
        all_data_test = np.array(all_data_test)
        all_result_test = np.array(all_result_test)
        dtrain = xgb.DMatrix(all_data, label=all_result)
        dtest = xgb.DMatrix(all_data_test, label=all_result_test)

        params = {
            'objective': 'reg:squarederror',
            'max_depth': 5,
            'learning_rate': 0.05,
            'silent': 1,
            'n_estimators': 300
        }
        early_stop = 10


        bst = xgb.train(params, dtrain, num_boost_round=300, early_stopping_rounds=early_stop, evals=[(dtest, 'test')])
        

        output_info = []
        test_cnt = 0
        error_cnt = 0
        y_pred = bst.predict(dtest)
        for i in range(len(y_pred)):
            test_cnt += 1
            if abs(all_result_test[i] - y_pred[i])/all_result_test[i] > 0.1:
                output_info.append("result:{:.4f}, y_pred:{:.4f}, diff:{:.4f}".format(all_result_test[i], y_pred[i], (all_result_test[i] - y_pred[i])/all_result_test[i]))
                error_cnt += 1
        
        

        train_cnt = 0
        train_error_cnt = 0
        y_pred = bst.predict(dtrain)
        for i in range(len(y_pred)):
            train_cnt += 1
            if abs(all_result[i] - y_pred[i])/all_result[i] > 0.1:
                output_info.append("result:{:.4f}, y_pred:{:.4f}, diff:{:.4f}".format(all_result[i], y_pred[i], (all_result[i] - y_pred[i])/all_result[i]))
                train_error_cnt += 1

        # for info in output_info:
        #     print(info)

        print("test_test: n_test:{}, n_correct:{}, accuracy:{:.3f}".format(test_cnt, test_cnt-error_cnt, (test_cnt-error_cnt)/test_cnt))
        print("train_test: n_test:{}, n_correct:{}, accuracy:{:.3f}".format(train_cnt, train_cnt-train_error_cnt, (train_cnt-train_error_cnt)/train_cnt))
        print("total: n_test:{}, n_correct:{}, accuracy:{:.3f}".format(train_cnt + test_cnt, train_cnt + test_cnt - error_cnt - train_error_cnt, (train_cnt + test_cnt - error_cnt - train_error_cnt)/(train_cnt + test_cnt)))
        accuracy_dict[key] = [round((test_cnt-error_cnt)/test_cnt, 3), round((train_cnt-train_error_cnt)/train_cnt, 3), round((train_cnt + test_cnt - error_cnt - train_error_cnt)/(train_cnt + test_cnt), 3)]
        if if_save:
            predict_model_file_name = "kernel_latency/{}_kernel_predictor_{}.bst".format(key, args.batch_size)
            bst.save_model(predict_model_file_name)

    acc_list = ["test_acc", "train_acc", "total_acc"]
    for key in accuracy_dict:
        for idx, acc in enumerate(accuracy_dict[key]):
            print("{}_{}: {}".format(key, acc_list[idx], acc))
    
    print("total_n_data:", total_n_data)


class latency_perdictor:
    def __init__(self, args):
        self.all_op_list =  ["fill_null", "sigrid_hash", "bucketize", "logit", "firstx", "boxcox", "clamp", "onehot", "ngram", "mapid"]
        self.args = args
        # self.model_dict = {
        #     "1d": ["fill_null", "sigrid_hash", "bucketize", "logit", "boxcox", "clamp", "mapid"],
        #     "firstx": ["firstx"], 
        #     "ngram": ["ngram"], 
        #     "onehot": ["onehot"]
        # }
        self.model_dict = {
            "1d": ["fill_null", "sigrid_hash", "logit", "boxcox", "clamp", "mapid"],
            "bucketize": ["bucketize"],
            "firstx": ["firstx"], 
            "ngram": ["ngram"], 
            "onehot": ["onehot"]
        }
        self.addr = "/workspace/RAP/RAP_end_to_end/utils/"
        self.one_dim_model = xgb.Booster()
        self.bucketize_model = xgb.Booster()
        self.firstx_model = xgb.Booster()
        self.ngram_model = xgb.Booster()
        self.onehot_model = xgb.Booster()

        self.base_batch_size = 4096
        self.one_dim_model.load_model(self.addr + "kernel_latency/1d_kernel_predictor_{}.bst".format(self.base_batch_size))
        self.bucketize_model.load_model(self.addr + "kernel_latency/bucketize_kernel_predictor_{}.bst".format(self.base_batch_size))
        self.firstx_model.load_model(self.addr + "kernel_latency/firstx_kernel_predictor_{}.bst".format(self.base_batch_size))
        self.ngram_model.load_model(self.addr + "kernel_latency/ngram_kernel_predictor_{}.bst".format(self.base_batch_size))
        self.onehot_model.load_model(self.addr + "kernel_latency/onehot_kernel_predictor_{}.bst".format(self.base_batch_size))


        # load scaling factor:
        self.scaling_factor_dict = {}
        with open(self.addr + "kernel_latency/op_scaling_factor_dict.pkl", "rb") as f:
            self.scaling_factor_dict = pickle.load(f)


    def choose_model(self, op):
        if op in self.model_dict["1d"]:
            return self.one_dim_model
        elif op == "bucketize":
            return self.bucketize_model
        elif op == "firstx":
            return self.firstx_model
        elif op == "ngram":
            return self.ngram_model
        elif op == "onehot":
            return self.onehot_model
        else:
            print("op:{} not in model_dict".format(op))
            return -1            

    def predict_single(self, op, width, param):
        if op == "fill_null_dense" or op == "fill_null_sparse":
            op = "fill_null"
        model = self.choose_model(op)
        op_id = self.all_op_list.index(op)

        if op in self.model_dict["1d"]:
            input_data = np.array([op_id, width, 1]).reshape(1, -1)
        elif op == "bucketize":
            input_data = np.array([op_id, 1, width]).reshape(1, -1)
        elif op == "firstx":
            input_data = np.array([op_id, 1, min(width, param)]).reshape(1, -1)
        elif op == "ngram":
            true_width = (width - min(param, width) + 1) * min(param, width)
            input_data = np.array([op_id, 1, true_width]).reshape(1, -1)
        elif op == "onehot":
            input_data = np.array([op_id, 1, width]).reshape(1, -1)
        
        latency = model.predict(xgb.DMatrix(input_data))
        return round(latency[0], 4)

    def predict_list(self, op, nBatch, width_list, param_list):
        if op == "fill_null_dense" or op == "fill_null_sparse":
            op = "fill_null"
        model = self.choose_model(op)
        op_id = self.all_op_list.index(op)

        if nBatch != len(width_list) or nBatch != len(param_list):
            print("op:{} nBatch:{} != len(width_list):{} or nBatch:{} != len(param_list):{}".format(op, nBatch, len(width_list), nBatch, len(param_list)))
            print("nBatch:{} != len(width_list):{} or nBatch:{} != len(param_list):{}".format(nBatch, len(width_list), nBatch, len(param_list)))
            return -1

        if op in self.model_dict["1d"]:
            total_width = sum(width_list)
            input_data = np.array([op_id, total_width, 1]).reshape(1, -1)
        elif op == "bucketize":
            total_width = sum(width_list)
            avg_width = total_width / nBatch
            input_data = np.array([op_id, nBatch, avg_width]).reshape(1, -1)
        elif op == "firstx":
            total_width = sum([min(width, param) for width, param in zip(width_list, param_list)])
            avg_width = total_width / nBatch
            input_data = np.array([op_id, nBatch, avg_width]).reshape(1, -1)
        elif op == "ngram":
            total_width = 0
            for width, param in zip(width_list, param_list):
                total_width += (width - min(param, width) + 1) * min(param, width)
            avg_width = total_width / nBatch
            input_data = np.array([op_id, nBatch, avg_width]).reshape(1, -1)
        elif op == "onehot":
            total_width = sum(width_list)
            avg_width = total_width / nBatch
            input_data = np.array([op_id, nBatch, avg_width]).reshape(1, -1)
        
        latency = model.predict(xgb.DMatrix(input_data))
        return round(latency[0], 4)

    def predict_fused_kernel(self, fused_kernel, nOp=None):
        lantency = 0
        ratio = 0
        if nOp == None:
            lantency = self.predict_list(fused_kernel.op_name, fused_kernel.nOp, fused_kernel.width_list, fused_kernel.param_list)
        else:
            lantency = self.predict_list(fused_kernel.op_name, nOp, fused_kernel.width_list[0:nOp], fused_kernel.param_list[0:nOp])
        
        if fused_kernel.batch_size == "1": # dense feature: args.batch_size
            ratio = int(self.args.batch_size / self.base_batch_size)
        elif fused_kernel.batch_size == "2": # sparse feature: args.batch_size * args.nDev
            ratio = int(self.args.batch_size * self.args.nDev / self.base_batch_size)
        else:
            print(fused_kernel.op_name, "Kernel Latency Predictor: fused_kernel.batch_size:{} not supported".format(fused_kernel.batch_size))
            exit(-1)
        
        op_name = fused_kernel.op_name
        if op_name == "fill_null_dense" or op_name == "fill_null_sparse":
            op_name = "fill_null"
        w_0, w_1 = self.scaling_factor_dict[op_name]
        scaling_factor = w_0 * ratio + w_1

        lantency = lantency * scaling_factor
        return lantency
    
    def predict_ngram_capacity(self, nBatch):
        model = self.ngram_model
        op_id = self.all_op_list.index("ngram")

        fuse_degree = nBatch
        op_width = (8 - 4 + 1) * 4
        input_data = np.array([op_id, fuse_degree, op_width]).reshape(1, -1)
        latency = model.predict(xgb.DMatrix(input_data))
        return round(latency[0], 4)


if __name__ == "__main__":
    args = get_args()

    # torch.manual_seed(42)
    # train_predicor(args, if_save=True)
    # train_data_prepare_predictor(args, if_save=True)


    # kernel_model = latency_perdictor(args)
    # data_prepare_model = data_prepare_perdictor(args)

    # print(kernel_model.predict_single("fill_null", 1, 1), data_prepare_model.predict_single("fill_null", 1, 1))
    # print(kernel_model.predict_single("ngram", 8, 4), data_prepare_model.predict_single("ngram", 8, 4))
    # print(kernel_model.predict_single("firstx", 8, 4), data_prepare_model.predict_single("firstx", 8, 4))

    # print(model.predict_ngram_capacity(128))

