torchx run -s local_cwd dist.ddp -j 1x2 --gpu 2 --script test_installation.py

# python BatchedFusedEmbedding_test.py --nDev=4

# python BreakBackwardTest.py

python Cuda_Preprocess_Kernel_Test.py

# python Cudf_load.py
echo "Installation Test Finished"