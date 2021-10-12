###
 # @Description: 
 # @Author: Li Siheng
 # @Date: 2021-10-12 01:46:32
 # @LastEditTime: 2021-10-12 02:05:55
### 
CUDA_VISIBLE_DEVICES=4 python main.py \
    --gpus 1 \
    --accelerator 'ddp' \
    --max_epochs 10 \
    --lr 1e-5 \
    --train_batchsize 16 \
    --valid_batchsize 16 \
    --num_workers 8 \
    --val_check_interval 0.5 \
    --data_dir './data/english/scientific' \
    --pretrain_model 'bert-base-uncased' \
    --model_name 'BertLSTMModel' \
    --rnn_size 256 \
    --rnn_nlayer 1 
