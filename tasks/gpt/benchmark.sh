cd $(dirname $0)

OMP_NUM_THREADS=128 torchrun --nproc_per_node=1 --master_port=1936 benchmark_gemini.py
