BATCH_SIZE=16
NUM_GPUS=7
n_embed=2048
num_iterations=80004
n_head=$(python -c "print(int($n_embed / 64))")
n_layer=20
echo "Running with n_embed = $n_embed, n_head = $n_head, n_layer = $n_layer, num_iterations = $num_iterations, learning_rate = 0.07, NUM_GPUS = $NUM_GPUS, BATCH_SIZE = $BATCH_SIZE"

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
torchrun --standalone --nproc-per-node=$NUM_GPUS train.py \
--run_name "muP_sweep_nanomdm_n_embed=$n_embed,n_head=$n_head,n_layer=$n_layer,n_iter=$num_iterations,HEBOSEARCH" \
--per_gpu_batch_size $BATCH_SIZE \
--global_batch_size $(($BATCH_SIZE * $NUM_GPUS)) \
--n_layer $n_layer \
--n_head $n_head \
--n_embed $n_embed \
--num_iterations $num_iterations \
--project_name "muP_sweep_nanomdm_exp" \
--tags "n_embed=$n_embed" \
--vres=True \
--warmdown_iters="30%" \
--learning_rate=0.14 \
--weight_decay=0.0099 \
--do_compile=True \
--lr_wtexweight=2.3297 \
--lr_attnxc_qxweight=0.8495 \
--lr_attnxc_kxweight=3.1112 \
--lr_attnxc_vxweight=1.3249 \
--lr_attnxc_projxweight=0.0741 \
--lr_mlpxc_fcxweight=2.8440 \
--lr_mlpxc_projxweight=0.6332 \
--lr_lm_headxweight=2.8089 \
--lr_lamb1=0.1269 \
--lr_lamb2=4.1834 \
--initstd_wtexweight=0.3376 \
--initstd_attnxc_qxweight=0.0906 \
--initstd_attnxc_kxweight=0.0589 \
--initstd_attnxc_vxweight=2.4990 \
--initstd_attnxc_projxweight=0.0 \
--initstd_mlpxc_fcxweight=0.5374 \
--initstd_mlpxc_projxweight=0.0 \
--initstd_lm_headxweight=0.0 \
--initstd_lamb1=1.4707 \
--initstd_lamb2=0.1110 