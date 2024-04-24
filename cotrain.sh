python main_cotrain.py --model coclr --net1 stftnet --net2 tinysleepnet_cnn --epochs 100 --seq_len 16 \
--dataset edf153_cotrain_freq \
--batch_size 64 \
--topk 5 \
--moco-k 2048 \
--seed 42 \
--data_path1 /workspace/leyu/sleepstage/sleep-cassette/npz \
--data_path2 /workspace/leyu/sleepstage/sleep-cassette/freq \
--pretrain /workspace/leyu/sleepstage/TimeSeriesCoCLR/log-pretrain/infonce_k2048_edf153_stft-dim128_stftnet_bs64_seq16_lr0.001/model/model_best.pth.tar /workspace/leyu/sleepstage/TimeSeriesCoCLR/log-copretrain/coclr-top5_k2048_edf153_cotrain_freq_tinysleepnet_cnn_stftnetbs64_lr0.001_len16_ds1/model/model_best.pth.tar \
--reverse