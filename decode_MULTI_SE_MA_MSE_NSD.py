# -*- coding: utf-8 -*-

from cProfile import label
import os
import numpy as np
import torch
import tqdm
import argparse

import HTK
import config
import utils
from model import MULTI_SE_MA_MSE_NSD
from reader import RTTM_to_Speaker_Mask


def load_ivector(speaker_embedding_txt):
    SCP_IO = open(speaker_embedding_txt)
    speaker_embedding = {}
    raw_lines = [l for l in SCP_IO]
    SCP_IO.close()
    speaker_embedding_list = []
    for i in range(len(raw_lines) // 2):
        speaker = raw_lines[2*i].split()[0]
        session = "-".join(speaker.split("-")[:-1])
        real_speaker = speaker.split("-")[-1]
        if session not in speaker_embedding.keys():
            speaker_embedding[session] = {}
        ivector = torch.from_numpy(np.array(raw_lines[2*i+1].split()[:-1], np.float32))
        speaker_embedding[session][real_speaker] = ivector
        speaker_embedding_list.append(ivector)
    return speaker_embedding, speaker_embedding_list


def load_htk(path):
    nSamples, sampPeriod, sampSize, parmKind, data = HTK.readHtk(path)
    htkdata = np.array(data).reshape(nSamples, int(sampSize / 4))
    #print(nSamples)
    return nSamples, htkdata

def load_single_channel_feature(file_path, window_len=800, hop_len=400):
    nSamples, htkdata = load_htk(file_path)
    # htkdata: T * F
    cur_frame, feature, intervals, total_frame = 0, [], [], nSamples
    while(cur_frame < total_frame):
        if cur_frame + window_len <= total_frame:
            feature.append(htkdata[cur_frame:cur_frame+window_len, : ])
            intervals.append((cur_frame, cur_frame+window_len))
            cur_frame += hop_len
        else:
            start = max(0, total_frame-window_len)
            feature.append(htkdata[start:total_frame, : ])
            intervals.append((start, total_frame))
            cur_frame += window_len
    return feature, intervals, total_frame

def preds_to_rttm(preds, intervals, dur, output_path):
    rttm = np.zeros([preds[0].shape[0], dur, preds[0].shape[2]])
    for i, p in enumerate(preds):
        rttm[:, intervals[i][0]: intervals[i][1], :] = p
    np.save(output_path, rttm)

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    torch.cuda.set_device(0)
    nnet = MULTI_SE_MA_MSE_NSD(config.configs[args.model_config])
    utils.load_checkpoint(nnet, None, args.model_path)
    nnet = nnet.cuda()
    nnet.eval()
    softmax = torch.nn.Softmax(dim=2)

    file_list = {}
    with open(args.feature_list) as INPUT:
        for l in INPUT:
            session = os.path.basename(l).split('.')[0]
            file_list[session] = l.rstrip()

    label_init = RTTM_to_Speaker_Mask(args.init_rttm)
    embedding, _ = load_ivector(args.embedding_list)
    _, train_set_speaker_embedding = load_ivector(args.train_set_speaker_embedding_list)
    idxs = list(range(len(train_set_speaker_embedding)))
    for session in tqdm.tqdm(file_list.keys()):
        speaker_embedding = []
        speaker_list = list(embedding[session].keys())
        num_speaker = len(speaker_list)
        for spk in embedding[session].keys():
            speaker_embedding.append(embedding[session][spk])
        for idx in np.random.choice(idxs, args.max_speaker - num_speaker, replace=False):
            speaker_embedding.append(train_set_speaker_embedding[idx])
        speaker_embedding = torch.stack(speaker_embedding) # num_speaker * embedding_dim
        feature, intervals, total_frame = load_single_channel_feature(file_list[session], args.max_utt_durance, int(args.max_utt_durance/4*3))
        output_path = os.path.join(args.output_dir, session)
        preds, i, cur_utt, batch, batch_intervals, new_intervals = [], 0, 0, [], [], []
        with torch.no_grad():
            for m in feature:
                batch.append(torch.from_numpy(m.astype(np.float32)))
                batch_intervals.append(intervals[cur_utt])
                cur_utt += 1
                i += 1
                if (i == args.batch_size) or (len(feature) == cur_utt):
                    length = [item.shape[0] for item in batch]
                    ordered_index = sorted(range(len(length)), key=lambda k: length[k], reverse = True)
                    Time, Freq = batch[ordered_index[0]].shape
                    cur_batch_size = len(length)
                    input_data = np.zeros([cur_batch_size, Time, Freq]).astype(np.float32)
                    mask_data = np.zeros([cur_batch_size, args.max_speaker, Time]).astype(np.float32)
                    nframes = []
                    batch_speaker_embedding = []
                    for i, id in enumerate(ordered_index):
                        input_data[i, :length[id], :] = batch[id]
                        nframes.append(length[id])
                        batch_speaker_embedding.append(speaker_embedding)
                        mask = label_init.get_mixture_utternce_label_informed_speaker(session, speaker_list, start=batch_intervals[id][0], end=batch_intervals[id][1], max_speaker=args.max_speaker)[..., 1]
                        if args.remove_overlap:
                            overlap = np.sum(mask, axis=0)
                            mask[:, overlap>=2] = 0
                        mask_data[i, :, :mask.shape[1]] = mask
                        new_intervals.append(batch_intervals[id])
                    input_data = torch.from_numpy(input_data).transpose(1, 2).cuda()
                    batch_speaker_embedding = torch.stack(batch_speaker_embedding).cuda() # B * num_speaker * embedding_dim
                    mask_data = torch.from_numpy(mask_data).cuda()
                    ypreds = nnet(input_data, batch_speaker_embedding, mask_data, nframes, args.split_seg)
                    ypreds = torch.stack([k for k in ypreds]) # speaker * T * 3
                    ypreds = softmax(ypreds).detach().cpu().numpy()
                    cur_frame = 0
                    for n in nframes:
                        #print(n)
                        preds.append(ypreds[:num_speaker, cur_frame:(cur_frame+n), :])
                        cur_frame += n
                    i, batch, batch_intervals = 0, [], []
        preds_to_rttm(preds, new_intervals, total_frame, output_path)

def make_argparse():
    # Set up an argument parser.
    parser = argparse.ArgumentParser(description='Prepare ivector extractor weights for ivector extraction.')
    parser.add_argument('--embedding_list', metavar='PATH', required=True,
                        help='embedding_list.')
    parser.add_argument('--train_set_speaker_embedding_list', metavar='PATH', required=True,
                        help='train_set_speaker_embedding_list.')
    parser.add_argument('--feature_list', metavar='PATH', required=True,
                        help='feature_list')
    parser.add_argument('--model_path', metavar='PATH', required=True,
                        help='model_path.')  
    parser.add_argument('--output_dir', metavar='PATH', required=True,
                        help='output_dir.')                       
    parser.add_argument('--max_speaker', metavar='PATH', type=int, default=8,
                help='max_speaker.')
    parser.add_argument('--init_rttm', metavar='PATH', required=True,
                        help='init_rttm.')
    parser.add_argument('--model_config', metavar='PATH', type=str, default="configs_4Speakers_ivectors128_2Classes",
                help='domain_list.')
    parser.add_argument('--max_utt_durance', metavar='PATH', type=int, default=800*32,
                help='max_utt_durance.')
    parser.add_argument('--split_seg', metavar='PATH', type=int, default=800,
                help='split_seg.')
    parser.add_argument('--batch_size', metavar='PATH', type=int, default=8,
                help='batch_size.')
    parser.add_argument('--remove_overlap', action="store_true", 
                help='remove_overlap.')      
    return parser


if __name__ == '__main__':
    parser = make_argparse()
    args = parser.parse_args()
    main(args)
