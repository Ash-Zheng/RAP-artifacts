import torch
import torch.nn as nn
from torchrec.distributed.shard import shard
from torchrec.distributed.types import (
    ParameterSharding,
    ShardingPlan,
    EnumerableShardingSpec,
    ShardMetadata,
    ShardingEnv,
    ShardingType
)


class SignalLayer(nn.Module):
    def __init__(self, input_queue):
        super(SignalLayer, self).__init__()

        self.signal_layer = nn.Identity()
    
        self.recorder_1 = torch.cuda.Event(enable_timing=True)
        self.recorder_2 = torch.cuda.Event(enable_timing=True)

        self.queue = input_queue
    
    def forward(self, input):
        return self.signal_layer(input)


class ShardedEmbeddingTable(nn.Module):
    def __init__(self, embedding_tables, plan, device, group, dim, batch_size, table_names, input_queue=None):
        super(ShardedEmbeddingTable, self).__init__()
        
        self.dim = dim
        self.table_names = table_names
        self.batch_size = batch_size

        self.sharded_emts = shard(
            module=embedding_tables,
            env=ShardingEnv.from_process_group(group),
            plan=plan.get_plan_for_module(""),
            device=device
        )

        self.signal_layer = SignalLayer(input_queue)



    def forward(self, sparse_input):
        sharded_model_pred_jts = self.sharded_emts(sparse_input).wait()

        B = sparse_input.stride() # batch_size

        sparse = sharded_model_pred_jts.to_dict()
        sparse_values = []
        for name in self.table_names:
            sparse_values.append(sparse[name])

        result = torch.cat(sparse_values, dim=1).reshape(self.batch_size, len(self.table_names), self.dim)

        # return result
        return self.signal_layer(result)

    def input_comm(self, sparse_input):
        dist_input = self.sharded_emts.input_comm(sparse_input)

        return dist_input

    def forward_on_dist_input(self, dist_input):
        sharded_model_pred_jts = self.sharded_emts.forward_without_input_comm(dist_input).wait()

        sparse = sharded_model_pred_jts.to_dict()
        sparse_values = []
        for name in self.table_names:
            sparse_values.append(sparse[name])

        result = torch.cat(sparse_values, dim=1).reshape(self.batch_size, len(self.table_names), self.dim)

        # return result
        return self.signal_layer(result)