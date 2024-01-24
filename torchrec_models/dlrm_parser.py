import argparse

def get_args():
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8192, help="local batch size to use for training"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="number of dataloader workers",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=30000001,
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to 100_000 if num_embeddings_per_feature is not supplied.",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        # default=None,
        default="45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35",
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    # test_tables: "65536,65536,65536,65536,65536,65536,65536,65536,65536,65536,65536,65536,65536,65536,65536,65536,65536,65536,65536,65536,65536,65536,65536,65536,65536,65536"
    # Reduced_Terabyte: "10000000,36746,17245,7413,20243,3,7114,1441,62,10000000,1572176,345138,10,2209,11267,128,4,974,14,10000000,10000000,10000000,452104,12606,104,35"
    # Terabyte: "45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35"
    # Kaggle: "1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"

    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="1024,1024,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--undersampling_rate",
        type=float,
        help="Desired proportion of zero-labeled samples to retain (i.e. undersampling zero-labeled rows)."
        " Ex. 0.3 indicates only 30pct of the rows with label 0 will be kept."
        " All rows with label 1 will be kept. Value should be between 0 and 1."
        " When not supplied, no undersampling occurs.",
    )
    parser.add_argument(
        "--seed",
        type=float,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        default=True,
        help="Use pinned memory when loading data.",
    )
    parser.add_argument(
        "--in_memory_binary_criteo_path",
        type=str,
        # default="/home/yuke_wang/zheng/torchrec/processed",
        default=None,
        help="Path to a folder containing the binary (npy) files for the Criteo dataset."
        " When supplied, InMemoryBinaryCriteoIterDataPipe is used.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate.",
    )
    parser.add_argument(
        "--shuffle_batches",
        type=bool,
        default=False,
        help="Shuffle each batch during training.",
    )
    parser.add_argument(
        "--nDev",
        type=int,
        default=4,
        help="number of GPUs",
    )
    parser.add_argument(
        "--single_table",
        type=bool,
        default=False,
        help="if_single_table",
    )
    parser.add_argument(
        "--preprocessing_op",
        type=str,
        default="fill_null", # "fill_null", "sigrid_hash", "bucketize", "logit", "firstx", "boxcox", "clamp", "onehot", "ngram", "mapid" 
        help="if_single_table",
    )
    parser.add_argument(
        "--dlrm_layer",
        type=str,
        default="mlp_fwd", # "emb_fwd", "mlp_fwd", "mlp_bwd", "emb_bwd", "grad_comm"
        help="dlrm_layer for capcacity test",
    )
    parser.add_argument(
        "--nDense",
        type=int,
        default=13, 
        help="nDense",
    )
    parser.add_argument(
        "--nSparse",
        type=int,
        default=26, 
        help="nSparse",
    )
    parser.add_argument(
        "--retrain_model",
        type=bool,
        default=False, 
        help="retrain_model for prediction model",
    )
    parser.add_argument(
        "--hotness",
        type=int,
        default=1, 
        help="hotness",
    )
    parser.add_argument(
        "--hotness_list",
        type=str,
        default=None, 
        help="hotnes_list",
    )
    parser.add_argument(
        "--fixed_table_length",
        type=int,
        default=0, 
        help="fixed_table_length",
    )
    parser.add_argument(
        "--op_width",
        type=int,
        default=1, 
        help="op_width",
    )
    parser.add_argument(
        "--fuse_degree",
        type=int,
        default=1, 
        help="fuse_degree",
    )
    parser.add_argument(
        "--preprocessing_plan",
        type=int,
        default=0, 
        help="preprocessing_plan",
    )
    parser.add_argument(
        "--max_width",
        type=int,
        default=256, 
        help="max_width",
    )
    parser.add_argument(
        "--nPartition",
        type=int,
        default=1,
        help="nPartition",
    )
    parser.add_argument(
        "--nWorker",
        type=int,
        default=8,
        help="nWorker",
    )
    parser.add_argument(
        "--MPS",
        type=bool,
        default=False,
        help="if MPS enabled",
    )


    args = parser.parse_args()

    args.dense_arch_layer_sizes += str(args.embedding_dim)
    args.num_embeddings_per_feature = list(
        map(int, args.num_embeddings_per_feature.split(","))
    )
    args.dense_arch_layer_sizes = list(
        map(int, args.dense_arch_layer_sizes.split(","))
    )
    args.over_arch_layer_sizes = list(
        map(int, args.over_arch_layer_sizes.split(","))
    )
    if args.hotness_list is not None:
        args.hotness_list = list(
            map(int, args.hotness_list.split(","))
        )

    args.cat_name = ["cat_" + str(i) for i in range(args.nSparse)]
    args.int_name = ["int_" + str(i) for i in range(args.nDense)]

    if args.fixed_table_length > 0:
        args.num_embeddings_per_feature = [args.fixed_table_length] * args.nSparse
    
    return args
