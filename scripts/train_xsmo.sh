DATA_PATH=~/train_data/mambaout/mini_imagenet100
CODE_PATH=~/dev/snn/spike-mambaout # modify code path here


ALL_BATCH_SIZE=256
NUM_GPU=4
GRAD_ACCUM_STEPS=2 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS


MODEL=mambaout_pico
DROP_PATH=0.025
DROP_PATH=0.025


cd $CODE_PATH && sh distributed_xsmo.sh $NUM_GPU $DATA_PATH \
--model $MODEL --opt adamw --lr 2.5e-4 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path $DROP_PATH