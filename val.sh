DATA_PATH=~/train_data/mambaout/mini_imagenet100
MODEL_TYPE=xsmo
CKPT=ckpt/xsmo_ckpt.tar

python validate.py $DATA_PATH $MODEL_TYPE \
    --model mambaout_pico \
    -b 256 \
    --pretrained \
    --checkpoint $CKPT \
    --num-gpu 1 