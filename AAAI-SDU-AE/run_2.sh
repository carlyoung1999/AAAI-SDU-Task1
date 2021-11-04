###
 # @Description: 
 # @Author: Li Siheng
 # @Date: 2021-10-12 01:46:32
 # @LastEditTime: 2021-10-27 06:55:16
### 
CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --gpus 2 \
    --accelerator 'ddp' \
    --max_epochs 10 \
    --bert_lr 1e-5 \
    --lr 1e-4 \
    --train_batchsize 16 \
    --valid_batchsize 16 \
    --num_workers 8 \
    --val_check_interval 0.25 \
    --data_dir './data/english/scientific' \
    --pretrain_model 'bert-base-uncased' \
    --model_name 'BertLSTMModel' \
    --use_crf \
    --rnn_size 256 \
    --rnn_nlayer 1 \
    --divergence 'js' \
    --adv_alpha 1.0 \
