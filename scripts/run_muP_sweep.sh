

BATCH_SIZE=16
NUM_GPUS=7

log_lrs=(-6)
n_embeds=(2048 1024 512 256)

for n_embed in ${n_embeds[@]}; do
    for log_lr in ${log_lrs[@]}; do
        lr=$(python -c "print(2** $log_lr)")
        num_iterations=$(python -c "print(int(40240 * 2048 / $n_embed))")
        n_head=$(python -c "print(int($n_embed / 64))")
        n_layer=$(python -c "print(int($n_embed / 128))")
        echo "Running with lr = $lr, n_embed = $n_embed, n_head = $n_head, n_layer = $n_layer"

        export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
        torchrun --standalone --nproc-per-node=$NUM_GPUS train.py \
            --run_name "muP_sweep_nanomdm_n_embed=$n_embed,lr=$lr,n_head=$n_head,n_layer=$n_layer,n_iter=$num_iterations" \
            --per_gpu_batch_size $BATCH_SIZE \
            --global_batch_size $(($BATCH_SIZE * $NUM_GPUS)) \
            --n_layer $n_layer \
            --n_head $n_head \
            --n_embed $n_embed \
            --learning_rate $lr \
            --num_iterations $num_iterations \
            --project_name "muP_sweep_nanomdm_exp" \
            --tags "n_embed=$n_embed,lr=$lr" \
            --vres=True \
            --warmdown_iters="10%"
    done
done
