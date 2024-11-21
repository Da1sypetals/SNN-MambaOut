NUM_PROC=$1
shift

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.launch \
  --use-env \
  --master_port=25641 \
  --nproc_per_node=$NUM_PROC \
  train_mambaout.py "$@"
