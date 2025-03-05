
BATCH_SIZE=32
NUM_GPUS=2

torchrun --standalone --nproc-per-node=$NUM_GPUS train.py \
    --run_name test \
    --per_gpu_batch_size $BATCH_SIZE \
    --global_batch_size $(($BATCH_SIZE * $NUM_GPUS)) \
    --n_layer 12 \
    --n_head 12 \
    --n_embed 768
