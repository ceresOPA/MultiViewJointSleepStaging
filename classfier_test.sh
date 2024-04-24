python main_classfier.py --net tinysleepnet_cnn --dataset isruc \
--batch_size 32 \
--data_path /workspace/leyu/sleepstage/ISRUC-Sleep/npz \
--seed 42 \
--train_what all \
--test /workspace/leyu/sleepstage/TimeSeriesCoCLR/log-eval-linclr/isruc_tinysleepnet_cnn_Adam_bs32_lr0.001_dp0.9_wd0.001_seqlen16_ds1_train-all_pt=-workspace-leyu-sleepstage-TimeSeriesCoCLR-log-pretrain-infonce_k2048_edf153_signal-dim128_tinysleepnet_cnn_bs64_seq16_lr0.001-model-model_best.pth.tar/model/model_best.pth.tar