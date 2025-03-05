

BATCH_SIZE=32
NUM_GPUS=8

log_lrs=(-12 -10 -8 -6 -4 -2)
n_embeds=(128 256 512 1024)

for n_embed in ${n_embeds[@]}; do
    for log_lr in ${log_lrs[@]}; do
        lr=$(python -c "print(2** $log_lr)")
        
        n_head=$(python -c "print(int($n_embed / 64))")
        echo "Running with lr = $lr, n_embed = $n_embed, n_head = $n_head"

        torchrun --standalone --nproc-per-node=$NUM_GPUS train.py \
            --run_name test \
            --per_gpu_batch_size $BATCH_SIZE \
            --global_batch_size $(($BATCH_SIZE * $NUM_GPUS)) \
            --n_layer 12 \
            --n_head $n_head \
            --n_embed $n_embed \
            --learning_rate $lr \
            --project_name "muP_sweep_nanomdm" \
            --tags "n_embed=$n_embed,lr=$lr"
        
    done
done
