# -*- coding: utf-8 -*-

import os
import sys
from train_Pretrain_SE import Train
from model import MULTI_SE_MA_MSE_NSD
from config import configs3_4Speakers_ivector_ivector128_xvectors128_2Classes
import torch
from loss_function import SoftCrossEntropy_SingleTargets

from reader import collate_fn_mask
from reader import Fbank_Embedding_Label_Mask, RTTM_to_Speaker_Mask


data="AMI_Headset"
feature_scp = f"data/{data}/train_cmn_slide_fbank_htk.list"
ivector_path = f"exp/nnet3_recipe_ivector/ivectors_AMI_Headset_train/ivectors_spk.txt"
oracle_rttm = f"data/{data}/train.rttm"

print(feature_scp)
print(ivector_path)
print(oracle_rttm)

max_utt_durance = 800
batchsize = 48
split_seg = -1
mixup_rate=0.5

label_2classes = RTTM_to_Speaker_Mask(oracle_rttm, differ_silence_inference_speech = False)

multiple_4speakers_2classes = Fbank_Embedding_Label_Mask(feature_scp, ivector_path, label_2classes, append_speaker=True, min_speaker=2, max_speaker=4, max_utt_durance=max_utt_durance, frame_shift=int(max_utt_durance/4*3), mixup_rate=mixup_rate, alpha=0.5)

output_dir = f"model/MULTI_SE_MA_MSE_NSD/Batchsize{batchsize}_4speakers_Segment{max_utt_durance}s_configs3_4Speakers_ivector128_xvectors128_2Classes_Mixup{mixup_rate}_{data}_reoccur"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
os.system("cp {} {}/{}".format(os.path.abspath(sys.argv[0]), output_dir, sys.argv[0]))
optimizer = torch.optim.Adam
# pdb.set_trace()
train = Train(multiple_4speakers_2classes, collate_fn_mask, MULTI_SE_MA_MSE_NSD, configs3_4Speakers_ivector_ivector128_xvectors128_2Classes, "MULTI_SE_MA_MSE_NSD", output_dir, optimizer, SoftCrossEntropy_SingleTargets, batchsize=batchsize, accumulation_steps=[(0, 1)], lr=0.0001, split_seg=split_seg, start_epoch=0, end_epoch=50, cuda=[0, 1], num_workers=8)
train.train()
