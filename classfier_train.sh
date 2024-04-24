python main_classfier.py --net stftnet --dataset isruc_stft --epochs 100 --seq_len 16 \
--batch_size 32 \
--data_path /workspace/leyu/sleepstage/ISRUC-Sleep/freq \
--seed 42 \
--pretrain /workspace/leyu/sleepstage/TimeSeriesCoCLR/log-copretrain/coclr-top5-R_k2048_edf153_cotrain_freq_stftnet_tinysleepnet_cnnbs64_lr0.001_len16_ds1/model/model_best.pth.tar