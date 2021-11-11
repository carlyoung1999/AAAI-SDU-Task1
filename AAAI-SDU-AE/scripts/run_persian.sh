CUDA_VISIBLE_DEVICES=2 python main.py \
    --gpus 1 \
    --accelerator 'ddp' \
    --max_epochs 10 \
    --bert_lr 5e-5 \
    --lr 1e-4 \
    --train_batchsize 16 \
    --valid_batchsize 16 \
    --num_workers 8 \
    --val_check_interval 0.5 \
    --data_dir './data/persian' \
    --pretrain_model 'm3hrdadfi/bert-fa-base-uncased-wikinli' \
    --model_name 'BertLSTMModel' \
    --rnn_size 256 \
    --rnn_nlayer 1 \
    --adversarial \
    --divergence 'js' \
    --adv_alpha 0.5 \
    --adv_nloop 1 \