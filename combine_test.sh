python combine_test.py --net1 tinysleepnet_cnn --net2 stftnet --dataset1 isruc --dataset2 isruc_stft \
--batch_size 32 \
--data_path1 /workspace/leyu/sleepstage/ISRUC-Sleep/npz \
--data_path2 /workspace/leyu/sleepstage/ISRUC-Sleep/freq \
--seed 42 \
--test1  /workspace/leyu/sleepstage/TimeSeriesCoCLR/log-eval-linclr/11111_isruc_tinysleepnet_cnn_Adam_bs32_lr0.001_dp0.9_wd0.001_seqlen16_ds1_train-last_pt=-workspace-leyu-sleepstage-TimeSeriesCoCLR-log-copretrain-coclr-top5_k2048_edf153_cotrain_freq_tinysleepnet_cnn_stftnetbs64_lr0.001_len16_ds1-model-model_best.pth.tar/model/model_best.pth.tar \
--test2 /workspace/leyu/sleepstage/TimeSeriesCoCLR/log-eval-linclr/11111_isruc_stft_stftnet_Adam_bs32_lr0.001_dp0.9_wd0.001_seqlen16_ds1_train-last_pt=-workspace-leyu-sleepstage-TimeSeriesCoCLR-log-copretrain-coclr-top5-R_k2048_edf153_cotrain_freq_stftnet_tinysleepnet_cnnbs64_lr0.001_len16_ds1-model-model_best.pth.tar/model/model_best.pth.tar \