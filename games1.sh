#!/user/bin/env/ bash
python -u  Mynew_main.py \
 --data=Games \
 --gpu=1 \
 --epoch=20 \
 --hidden_size=50 \
 --batchSize=50 \
 --user_long=orgat\
 --user_short=att \
 --item_long=orgat \
 --item_short=att \
 --user_update=rnn \
 --item_update=rnn \
 --lr=0.001 \
 --l2=0.0001 \
 --layer_num=3 \
 --item_max_length=50 \
 --user_max_length=50 \
 --attn_drop=0.3 \
 --feat_drop=0.3 \
 --record \
 --model_record \
 --usejTime \
 --usexTime \
 --useTime \
 --useMin \
 --duibi \
 --compare 1 \
 --a 1
#  --useOld \
