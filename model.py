# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from resnet import ResNet101


class LSTM_Projection(nn.Module):
    def __init__(self, input_size, hidden_size, linear_dim, num_layers=1, bidirectional=True, dropout=0):
        super(LSTM_Projection, self).__init__()
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        self.forward_projection = nn.Linear(hidden_size, linear_dim)
        self.backward_projection = nn.Linear(hidden_size, linear_dim)
        self.relu = nn.ReLU(True)

    def forward(self, x, nframes):
        '''
        x: [batchsize, Time, Freq]
        nframes: [len_b1, len_b2, ..., len_bN]
        '''
        packed_x = nn.utils.rnn.pack_padded_sequence(x, nframes, batch_first=True)
        packed_x_1, hidden = self.LSTM(packed_x)
        x_1, l = nn.utils.rnn.pad_packed_sequence(packed_x_1, batch_first=True)
        forward_projection = self.relu(self.forward_projection(x_1[..., :self.hidden_size]))
        backward_projection = self.relu(self.backward_projection(x_1[..., self.hidden_size:]))
        # x_2: [batchsize, Time, linear_dim*2]
        x_2 = torch.cat((forward_projection, backward_projection), dim=2)
        return x_2


class CNN2D_BN_Relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(CNN2D_BN_Relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels) #(N,C,H,W) on C
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, bias=True):
        super(SeparableConv1d,self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size, stride, kernel_size//2, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels) #(N,C,L) on C
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MA_MSE(nn.Module):
    def __init__(self, fea_dim=20*128, n_heads=8, speaker_embedding_path=""):
        super(MA_MSE, self).__init__()

        self.n_heads = n_heads

        #Dictionary number of cluster * speaker_embedding_dim
        self.m = torch.from_numpy(np.load(speaker_embedding_path).astype(np.float32))
        self.N_clusters, Emb_dim = self.m.shape

        # Define matrices W (from audio feature) and U (from embedding)
        self.W = nn.Linear(fea_dim, n_heads)
        self.U = nn.Linear(Emb_dim, n_heads)

        self.v = nn.Linear(n_heads, 1)
    
    def forward(self, x, mask):
        '''
        x: Batch * Fea * Time
        mask: Batch * speaker * Time
        '''
        Batch, Fea, Time = x.shape
        num_speaker = mask.shape[1]
        #x_1: [Batch, num_speaker, Time, Fea]
        x_1 = x.repeat(1, num_speaker, 1).reshape(Batch, num_speaker, Fea, Time).transpose(2, 3)

        #x_2: Average [Batch, num_speaker, Fea]
        x_2 = torch.sum(x_1 * mask[..., None], axis=2) / (1e-10 + torch.sum(mask, axis=2)[..., None])

        #self.W(x_2) [Batch, num_speaker, n_heads]
        w = self.W(x_2).repeat(1, self.N_clusters, 1).reshape(Batch, self.N_clusters, num_speaker, self.n_heads).transpose(1, 2)

        #self.U(self.m) [N_clusters, n_heads]
        m = self.m.cuda()
        u = self.U(m).repeat(Batch*num_speaker, 1).reshape(Batch, num_speaker, self.N_clusters, self.n_heads)

        #c: Attention [Batch, num_speaker, N_clusters]
        c = self.v(torch.tanh(w + u)).squeeze(dim=3)

        #a: normalized attention values [Batch, num_speaker, N_clusters]
        a = torch.sigmoid(c)

        #e: weighted sum of the vectors [Batch, num_speaker, Emb_dim]
        #[Batch, num_speaker, N_clusters, 1] * [1, 1, N_clusters, Emb_dim]
        e = torch.sum(a[..., None] * m[None, None, ...], dim=2)

        return e
    



class MA_MSE_NSD(nn.Module):
    '''
    INPUT (MFCC)
    IDCT: MFCC to FBANK
    Batchnorm
    Stats pooling: batchnorm-cmn
    Combine inputs:  (batchnorm, batchnorm-cmn Speaker Detection Block
    4 layers CNN: Conv2d (in channels, out channels, kernel size, stride=1, padding=0 dilation=1, groups=1, bias=True, padding mode=zeros) 1 layer Splice-embedding:  (Convld SD, ivector-k) Linear layer
    2 layers Shared-blstm
    Attention
    1 layers CNN: Convld (in channels, out channels, kernel size, stride=1, padding=0 dilation=1, groups=1, bias=True, padding mode=zeros) Attention layer
    1 layers Combine-speaker
    1 layers BLSTM
    1 layers Dense: 4 dependent FC layers
    '''
    def __init__(self, configs):
        super(MA_MSE_NSD, self).__init__()
        self.input_size = configs["input_dim"]
        self.cnn_configs = configs["cnn_configs"]
        self.speaker_embedding_size = configs["speaker_embedding_dim"]
        self.Linear_dim = configs["Linear_dim"]
        self.Shared_BLSTM_size = configs["Shared_BLSTM_dim"]
        self.Linear_Shared_layer1_dim = configs["Linear_Shared_layer1_dim"]
        self.Linear_Shared_layer2_dim = configs["Linear_Shared_layer2_dim"]
        self.BLSTM_size = configs["BLSTM_dim"]
        self.BLSTM_Projection_dim = configs["BLSTM_Projection_dim"]
        self.output_size = configs["output_dim"]
        self.output_speaker = configs["output_speaker"]

        self.batchnorm = nn.BatchNorm2d(1)
        self.average_pooling = nn.AvgPool1d(configs["average_pooling"], stride=1, padding=configs["average_pooling"]//2)
        # MA-MSE
        self.mamse = MA_MSE(fea_dim=configs["fea_dim"], n_heads=configs["n_heads"], speaker_embedding_path=configs["embedding_path"])

        # Speaker Detection Block
        self.Conv2d_SD = nn.Sequential()
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD1', CNN2D_BN_Relu(self.cnn_configs[0][0], self.cnn_configs[0][1], self.cnn_configs[0][2], self.cnn_configs[0][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD2', CNN2D_BN_Relu(self.cnn_configs[1][0], self.cnn_configs[1][1], self.cnn_configs[1][2], self.cnn_configs[1][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD3', CNN2D_BN_Relu(self.cnn_configs[2][0], self.cnn_configs[2][1], self.cnn_configs[2][2], self.cnn_configs[2][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD4', CNN2D_BN_Relu(self.cnn_configs[3][0], self.cnn_configs[3][1], self.cnn_configs[3][2], self.cnn_configs[3][3]))

        self.splice_size = configs["splice_size"]
        self.Linear = nn.Linear(self.splice_size, self.Linear_dim)
        self.relu = nn.ReLU(True)
        self.Shared_BLSTMP_1 = LSTM_Projection(input_size=self.Linear_dim, hidden_size=self.Shared_BLSTM_size, linear_dim=self.Linear_Shared_layer1_dim, num_layers=1, bidirectional=True, dropout=0)
        self.Shared_BLSTMP_2 = LSTM_Projection(input_size=self.Linear_Shared_layer1_dim*2, hidden_size=self.Shared_BLSTM_size, linear_dim=self.Linear_Shared_layer2_dim, num_layers=1, bidirectional=True, dropout=0)

        self.conbine_speaker_size = self.Linear_Shared_layer2_dim * 2 * self.output_speaker
        self.BLSTMP = LSTM_Projection(input_size=self.conbine_speaker_size, hidden_size=self.BLSTM_size, linear_dim=self.BLSTM_Projection_dim, num_layers=1, bidirectional=True, dropout=0)
        FC = {}
        for i in range(self.output_speaker):
            FC[str(i)] = nn.Linear(self.BLSTM_Projection_dim*2, self.output_size)
        self.FC = nn.ModuleDict(FC)

    def forward(self, x, mask, nframes, split_seg=-1):
        '''
        x: Batch * Freq * Time
        mask : Batch * speaker * Time
        Batch * speaker(4) * Embedding
        nframe: descend order
        split: split long sequence to shorter segments to accelerate BLSTM training
                Time % split_seg == 0
        '''
        batchsize, Freq, Time = x.shape
        embedding_dim = self.speaker_embedding_size

        if type(nframes) == torch.Tensor:
            nframes = list(nframes.detach().cpu().numpy())

        # ************batchnorm statspooling*****************
        #batchnorm [batchsize, Freq, Time] -> [ batchsize, 1, Freq, Time]
        x_3 = self.batchnorm(x.reshape(batchsize, 1, Freq, Time)).squeeze(dim=1)

        x_3_mean = self.average_pooling(x_3)
        #(batchnorm, batchnorm_cmn): 2 * [batchsize, Freq, Time] -> [batchs ize, 2, Freq, Time]
        x_4 = torch.cat((x_3, x_3_mean), dim=1).reshape(batchsize, 2, Freq, Time)
        #(batchnorm, batchnorm_cmn) -> (bn_d0, bnc_d0, bn_d1, bnc_d1,..., bn_d40, bnc_d40)

        # **************CNN*************
        # [batchsize, Freq, Time] => [batchsize, Conv-4-out-filters, Freq, Time]
        x_5 = self.Conv2d_SD(x_4)

        #[batchsize, Conv-4-out-filters, Freq, Time] => [ batchsize, Conv-4-out-filters*Freq, Time ]
        x_6 = x_5.reshape(batchsize, -1, Time)
        Freq = x_6.shape[1]
        
        embedding = self.mamse(x_6, mask) # [Batch, num_speaker, Emb_dim]

        x_6_reshape = x_6.repeat(1, self.output_speaker, 1).reshape(batchsize * self.output_speaker, Freq, Time)
        #embedding: Batch * speaker * Embedding -> (Batch * speaker) * Embedding * Time
        embedding_reshape = embedding.reshape(-1, embedding_dim)[..., None].expand(batchsize * self.output_speaker, embedding_dim, Time)

        x_7 = torch.cat((x_6_reshape, embedding_reshape), dim=1)
        '''
        x_7: (Batch * speaker) * (Freq + Embedding) * Time
        '''
        #**************Linear*************
        #(Batch * speaker) * (Freq + Embedding) * Time =>(Batch * speaker) * Time * Linear_dim
        x_8 = self.relu(self.Linear(x_7.transpose(1, 2)))
        if split_seg > 0: #batchsize * self.output_speaker, Freq, Time
            num_seg = Time//split_seg
            x_8 = x_8.reshape(batchsize*self.output_speaker, num_seg, split_seg, -1).reshape(batchsize*self.output_speaker*num_seg, split_seg, -1)
            lens = [split_seg for i in range(batchsize*self.output_speaker*num_seg)] 
        #Shared_BLSTMP_1 (Batch * speaker) * Time * Linear_dim =>(Batch * speaker) * Time * (Linear_Shared_layer1_dim * 2)
        else:
            lens = [n for n in nframes for i in range(self.output_speaker)] 
        x_9 = self.Shared_BLSTMP_1(x_8, lens)
        #Shared_BLSTMP_2 (Batch * speaker) * Time * (Linear_Shared_layer1_dim * 2) =ã (Batch * speaker) * Time * (Linear_Shared_layer2_dim * 2)
        x_10 = self.Shared_BLSTMP_2(x_9, lens)

        #Combine-Speaker: (Batch * speaker) * Time * (Linear_Shared_layer2_dim * 2) => Batch * Time * (speaker * Linear_Shared_layer2_dim * 2)
        if split_seg > 0:
            x_11 = x_10.reshape(batchsize, self.output_speaker, num_seg, split_seg, -1).permute(0, 2, 3, 1, 4).reshape(batchsize*num_seg, split_seg, -1)
            lens = [split_seg for i in range(batchsize*num_seg)] 
        else:
            x_11 = x_10.reshape(batchsize, self.output_speaker, Time, -1).transpose(1, 2).reshape(batchsize, Time, -1)
            lens = nframes
        '''
        batchsize * Time * (1stspeaker 2ndspeaker 3rdspeaker 4thspeaker)
        '''

        #BLSTM: Batch * Time * (speaker * Linear_Shared_layer2_dim * 2) => Batch * Time * (BLSTM_Projection_dim * 2)
        x_12 = self.BLSTMP(x_11, lens)

        if split_seg > 0:
            x_12 = x_12.reshape(batchsize, num_seg, split_seg, -1).reshape(batchsize, num_seg*split_seg, -1)
        #Dimension reduction: remove the padding frames; Batch * Time * (BLSTM_Projection_dim * 2) => (Batch * Time) * (BLSTM_Projection_dim * 2)
        lens = [k for i, m in enumerate(nframes) for k in range(i * Time, m + i * Time)]
        x_13 = x_12.reshape(batchsize * Time, -1)[lens, :]
        '''
        1stBatch(1st sentence) length1 * (BLSTM_size * 2)
        2ndBatch(2nd sentence) length2 * (BLSTM_size * 2)
        ...
        '''
        out = []
        for i in self.FC:
            out.append(self.FC[i](x_13))
        return out


class SE_MA_MSE_NSD(nn.Module):
    '''
    speaker embedding + mse
    INPUT (MFCC)
    IDCT: MFCC to FBANK
    Batchnorm
    Stats pooling: batchnorm-cmn
    Combine inputs:  (batchnorm, batchnorm-cmn Speaker Detection Block
    4 layers CNN: Conv2d (in channels, out channels, kernel size, stride=1, padding=0 dilation=1, groups=1, bias=True, padding mode=zeros) 1 layer Splice-embedding:  (Convld SD, ivector-k) Linear layer
    2 layers Shared-blstm
    Attention
    1 layers CNN: Convld (in channels, out channels, kernel size, stride=1, padding=0 dilation=1, groups=1, bias=True, padding mode=zeros) Attention layer
    1 layers Combine-speaker
    1 layers BLSTM
    1 layers Dense: 4 dependent FC layers
    '''
    def __init__(self, configs):
        super(SE_MA_MSE_NSD, self).__init__()
        self.input_size = configs["input_dim"]
        self.cnn_configs = configs["cnn_configs"]
        self.speaker_embedding_size = configs["speaker_embedding_dim"]
        self.Linear_dim = configs["Linear_dim"]
        self.Shared_BLSTM_size = configs["Shared_BLSTM_dim"]
        self.Linear_Shared_layer1_dim = configs["Linear_Shared_layer1_dim"]
        self.Linear_Shared_layer2_dim = configs["Linear_Shared_layer2_dim"]
        self.BLSTM_size = configs["BLSTM_dim"]
        self.BLSTM_Projection_dim = configs["BLSTM_Projection_dim"]
        self.output_size = configs["output_dim"]
        self.output_speaker = configs["output_speaker"]

        self.batchnorm = nn.BatchNorm2d(1)
        self.average_pooling = nn.AvgPool1d(configs["average_pooling"], stride=1, padding=configs["average_pooling"]//2)
        # MA-MSE
        self.mamse = MA_MSE(fea_dim=configs["fea_dim"], n_heads=configs["n_heads"], speaker_embedding_path=configs["embedding_path"])

        # Speaker Detection Block
        self.Conv2d_SD = nn.Sequential()
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD1', CNN2D_BN_Relu(self.cnn_configs[0][0], self.cnn_configs[0][1], self.cnn_configs[0][2], self.cnn_configs[0][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD2', CNN2D_BN_Relu(self.cnn_configs[1][0], self.cnn_configs[1][1], self.cnn_configs[1][2], self.cnn_configs[1][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD3', CNN2D_BN_Relu(self.cnn_configs[2][0], self.cnn_configs[2][1], self.cnn_configs[2][2], self.cnn_configs[2][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD4', CNN2D_BN_Relu(self.cnn_configs[3][0], self.cnn_configs[3][1], self.cnn_configs[3][2], self.cnn_configs[3][3]))

        self.splice_size = configs["splice_size"]
        self.Linear = nn.Linear(self.splice_size + self.speaker_embedding_size, self.Linear_dim)
        self.relu = nn.ReLU(True)
        self.Shared_BLSTMP_1 = LSTM_Projection(input_size=self.Linear_dim, hidden_size=self.Shared_BLSTM_size, linear_dim=self.Linear_Shared_layer1_dim, num_layers=1, bidirectional=True, dropout=0)
        self.Shared_BLSTMP_2 = LSTM_Projection(input_size=self.Linear_Shared_layer1_dim*2, hidden_size=self.Shared_BLSTM_size, linear_dim=self.Linear_Shared_layer2_dim, num_layers=1, bidirectional=True, dropout=0)

        self.conbine_speaker_size = self.Linear_Shared_layer2_dim * 2 * self.output_speaker
        self.BLSTMP = LSTM_Projection(input_size=self.conbine_speaker_size, hidden_size=self.BLSTM_size, linear_dim=self.BLSTM_Projection_dim, num_layers=1, bidirectional=True, dropout=0)
        FC = {}
        for i in range(self.output_speaker):
            FC[str(i)] = nn.Linear(self.BLSTM_Projection_dim*2, self.output_size)
        self.FC = nn.ModuleDict(FC)

    def forward(self, x, overall_embedding, mask, nframes, split_seg=-1, return_embedding=False):
        '''
        x: Batch * Freq * Time
        mask : Batch * speaker * Time
        Batch * speaker(4) * Embedding
        nframe: descend order
        split: split long sequence to shorter segments to accelerate BLSTM training
                Time % split_seg == 0
        '''
        batchsize, Freq, Time = x.shape

        if type(nframes) == torch.Tensor:
            nframes = list(nframes.detach().cpu().numpy())

        # ************batchnorm statspooling*****************
        #batchnorm [batchsize, Freq, Time] -> [ batchsize, 1, Freq, Time]
        x_3 = self.batchnorm(x.reshape(batchsize, 1, Freq, Time)).squeeze(dim=1)

        x_3_mean = self.average_pooling(x_3)
        #(batchnorm, batchnorm_cmn): 2 * [batchsize, Freq, Time] -> [batchs ize, 2, Freq, Time]
        x_4 = torch.cat((x_3, x_3_mean), dim=1).reshape(batchsize, 2, Freq, Time)
        #(batchnorm, batchnorm_cmn) -> (bn_d0, bnc_d0, bn_d1, bnc_d1,..., bn_d40, bnc_d40)

        # **************CNN*************
        # [batchsize, Freq, Time] => [batchsize, Conv-4-out-filters, Freq, Time]
        x_5 = self.Conv2d_SD(x_4)

        #[batchsize, Conv-4-out-filters, Freq, Time] => [ batchsize, Conv-4-out-filters*Freq, Time ]
        x_6 = x_5.reshape(batchsize, -1, Time)
        Freq = x_6.shape[1]
        
        embedding = self.mamse(x_6, mask) # [Batch, num_speaker, Emb_dim]

        x_6_reshape = x_6.repeat(1, self.output_speaker, 1).reshape(batchsize * self.output_speaker, Freq, Time)
        #embedding: Batch * speaker * Embedding -> (Batch * speaker) * Embedding * Time
        embedding_dim1 = embedding.shape[2]
        embedding_reshape1 = embedding.reshape(-1, embedding_dim1)[..., None].expand(batchsize * self.output_speaker, embedding_dim1, Time)
        embedding_dim2 = overall_embedding.shape[2]
        embedding_reshape2 = overall_embedding.reshape(-1, embedding_dim2)[..., None].expand(batchsize * self.output_speaker, embedding_dim2, Time)

        x_7 = torch.cat((x_6_reshape, embedding_reshape1, embedding_reshape2), dim=1)
        '''
        x_7: (Batch * speaker) * (Freq + Embedding) * Time
        '''
        #**************Linear*************
        #(Batch * speaker) * (Freq + Embedding) * Time =>(Batch * speaker) * Time * Linear_dim
        x_8 = self.relu(self.Linear(x_7.transpose(1, 2)))
        if split_seg > 0: #batchsize * self.output_speaker, Freq, Time
            num_seg = Time//split_seg
            x_8 = x_8.reshape(batchsize*self.output_speaker, num_seg, split_seg, -1).reshape(batchsize*self.output_speaker*num_seg, split_seg, -1)
            lens = [split_seg for i in range(batchsize*self.output_speaker*num_seg)] 
        #Shared_BLSTMP_1 (Batch * speaker) * Time * Linear_dim =>(Batch * speaker) * Time * (Linear_Shared_layer1_dim * 2)
        else:
            lens = [n for n in nframes for i in range(self.output_speaker)] 
        x_9 = self.Shared_BLSTMP_1(x_8, lens)
        #Shared_BLSTMP_2 (Batch * speaker) * Time * (Linear_Shared_layer1_dim * 2) => (Batch * speaker) * Time * (Linear_Shared_layer2_dim * 2)
        x_10 = self.Shared_BLSTMP_2(x_9, lens)

        #Combine-Speaker: (Batch * speaker) * Time * (Linear_Shared_layer2_dim * 2) => Batch * Time * (speaker * Linear_Shared_layer2_dim * 2)
        if split_seg > 0:
            x_11 = x_10.reshape(batchsize, self.output_speaker, num_seg, split_seg, -1).permute(0, 2, 3, 1, 4).reshape(batchsize*num_seg, split_seg, -1)
            lens = [split_seg for i in range(batchsize*num_seg)] 
        else:
            x_11 = x_10.reshape(batchsize, self.output_speaker, Time, -1).transpose(1, 2).reshape(batchsize, Time, -1)
            lens = nframes
        '''
        batchsize * Time * (1stspeaker 2ndspeaker 3rdspeaker 4thspeaker)
        '''

        #BLSTM: Batch * Time * (speaker * Linear_Shared_layer2_dim * 2) => Batch * Time * (BLSTM_Projection_dim * 2)
        x_12 = self.BLSTMP(x_11, lens)

        if split_seg > 0:
            x_12 = x_12.reshape(batchsize, num_seg, split_seg, -1).reshape(batchsize, num_seg*split_seg, -1)
        #Dimension reduction: remove the padding frames; Batch * Time * (BLSTM_Projection_dim * 2) => (Batch * Time) * (BLSTM_Projection_dim * 2)
        lens = [k for i, m in enumerate(nframes) for k in range(i * Time, m + i * Time)]
        x_13 = x_12.reshape(batchsize * Time, -1)[lens, :]
        '''
        1stBatch(1st sentence) length1 * (BLSTM_size * 2)
        2ndBatch(2nd sentence) length2 * (BLSTM_size * 2)
        ...
        '''
        out = []
        for i in self.FC:
            out.append(self.FC[i](x_13))
        if return_embedding:
            return out, embedding
        else:
            return out


class MULTI_SE_MA_MSE_NSD(nn.Module):
    '''
    speaker embedding + mse
    INPUT (MFCC)
    IDCT: MFCC to FBANK
    Batchnorm
    Stats pooling: batchnorm-cmn
    Combine inputs:  (batchnorm, batchnorm-cmn Speaker Detection Block
    4 layers CNN: Conv2d (in channels, out channels, kernel size, stride=1, padding=0 dilation=1, groups=1, bias=True, padding mode=zeros) 1 layer Splice-embedding:  (Convld SD, ivector-k) Linear layer
    2 layers Shared-blstm
    Attention
    1 layers CNN: Convld (in channels, out channels, kernel size, stride=1, padding=0 dilation=1, groups=1, bias=True, padding mode=zeros) Attention layer
    1 layers Combine-speaker
    1 layers BLSTM
    1 layers Dense: 4 dependent FC layers
    '''
    def __init__(self, configs):
        super(MULTI_SE_MA_MSE_NSD, self).__init__()
        self.input_size = configs["input_dim"]
        self.cnn_configs = configs["cnn_configs"]
        self.Linear_dim = configs["Linear_dim"]
        self.Shared_BLSTM_size = configs["Shared_BLSTM_dim"]
        self.Linear_Shared_layer1_dim = configs["Linear_Shared_layer1_dim"]
        self.Linear_Shared_layer2_dim = configs["Linear_Shared_layer2_dim"]
        self.BLSTM_size = configs["BLSTM_dim"]
        self.BLSTM_Projection_dim = configs["BLSTM_Projection_dim"]
        self.output_size = configs["output_dim"]
        self.output_speaker = configs["output_speaker"]

        self.batchnorm = nn.BatchNorm2d(1)
        self.average_pooling = nn.AvgPool1d(configs["average_pooling"], stride=1, padding=configs["average_pooling"]//2)
        # MA-MSE
        self.mamse1 = MA_MSE(fea_dim=configs["fea_dim"], n_heads=configs["n_heads1"], speaker_embedding_path=configs["embedding_path1"])
        self.mamse2 = MA_MSE(fea_dim=configs["fea_dim"], n_heads=configs["n_heads2"], speaker_embedding_path=configs["embedding_path2"])
        # Speaker Detection Block
        self.Conv2d_SD = nn.Sequential()
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD1', CNN2D_BN_Relu(self.cnn_configs[0][0], self.cnn_configs[0][1], self.cnn_configs[0][2], self.cnn_configs[0][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD2', CNN2D_BN_Relu(self.cnn_configs[1][0], self.cnn_configs[1][1], self.cnn_configs[1][2], self.cnn_configs[1][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD3', CNN2D_BN_Relu(self.cnn_configs[2][0], self.cnn_configs[2][1], self.cnn_configs[2][2], self.cnn_configs[2][3]))
        self.Conv2d_SD.add_module('CNN2D_BN_Relu_SD4', CNN2D_BN_Relu(self.cnn_configs[3][0], self.cnn_configs[3][1], self.cnn_configs[3][2], self.cnn_configs[3][3]))

        self.splice_size = configs["splice_size"]
        self.Linear = nn.Linear(self.splice_size, self.Linear_dim)
        self.relu = nn.ReLU(True)
        self.Shared_BLSTMP_1 = LSTM_Projection(input_size=self.Linear_dim, hidden_size=self.Shared_BLSTM_size, linear_dim=self.Linear_Shared_layer1_dim, num_layers=1, bidirectional=True, dropout=0)
        self.Shared_BLSTMP_2 = LSTM_Projection(input_size=self.Linear_Shared_layer1_dim*2, hidden_size=self.Shared_BLSTM_size, linear_dim=self.Linear_Shared_layer2_dim, num_layers=1, bidirectional=True, dropout=0)

        self.conbine_speaker_size = self.Linear_Shared_layer2_dim * 2 * self.output_speaker
        self.BLSTMP = LSTM_Projection(input_size=self.conbine_speaker_size, hidden_size=self.BLSTM_size, linear_dim=self.BLSTM_Projection_dim, num_layers=1, bidirectional=True, dropout=0)
        FC = {}
        for i in range(self.output_speaker):
            FC[str(i)] = nn.Linear(self.BLSTM_Projection_dim*2, self.output_size)
        self.FC = nn.ModuleDict(FC)

    def forward(self, x, overall_embedding, mask, nframes, split_seg=-1, return_embedding=False):
        '''
        x: Batch * Freq * Time
        mask : Batch * speaker * Time
        Batch * speaker(4) * Embedding
        nframe: descend order
        split: split long sequence to shorter segments to accelerate BLSTM training
                Time % split_seg == 0
        '''
        batchsize, Freq, Time = x.shape
        if type(nframes) == torch.Tensor:
            nframes = list(nframes.detach().cpu().numpy())

        # ************batchnorm statspooling*****************
        #batchnorm [batchsize, Freq, Time] -> [ batchsize, 1, Freq, Time]
        x_3 = self.batchnorm(x.reshape(batchsize, 1, Freq, Time)).squeeze(dim=1)

        x_3_mean = self.average_pooling(x_3)
        #(batchnorm, batchnorm_cmn): 2 * [batchsize, Freq, Time] -> [batchs ize, 2, Freq, Time]
        x_4 = torch.cat((x_3, x_3_mean), dim=1).reshape(batchsize, 2, Freq, Time)
        #(batchnorm, batchnorm_cmn) -> (bn_d0, bnc_d0, bn_d1, bnc_d1,..., bn_d40, bnc_d40)

        # **************CNN*************
        # [batchsize, Freq, Time] => [batchsize, Conv-4-out-filters, Freq, Time]
        x_5 = self.Conv2d_SD(x_4)

        #[batchsize, Conv-4-out-filters, Freq, Time] => [ batchsize, Conv-4-out-filters*Freq, Time ]
        x_6 = x_5.reshape(batchsize, -1, Time)
        Freq = x_6.shape[1]
        
        embedding1 = self.mamse1(x_6, mask) # [Batch, num_speaker, Emb_dim]
        embedding2 = self.mamse2(x_6, mask) # [Batch, num_speaker, Emb_dim]
        
        x_6_reshape = x_6.repeat(1, self.output_speaker, 1).reshape(batchsize * self.output_speaker, Freq, Time)
        #embedding: Batch * speaker * Embedding -> (Batch * speaker) * Embedding * Time
        embedding_dim1 = embedding1.shape[2]
        embedding_reshape1 = embedding1.reshape(-1, embedding_dim1)[..., None].expand(batchsize * self.output_speaker, embedding_dim1, Time)

        embedding_dim2 = embedding2.shape[2]
        embedding_reshape2 = embedding2.reshape(-1, embedding_dim2)[..., None].expand(batchsize * self.output_speaker, embedding_dim2, Time)

        overall_embedding_dim = overall_embedding.shape[2]
        overall_embedding_reshape = overall_embedding.reshape(-1, overall_embedding_dim)[..., None].expand(batchsize * self.output_speaker, overall_embedding_dim, Time)

        x_7 = torch.cat((x_6_reshape, embedding_reshape1, embedding_reshape2, overall_embedding_reshape), dim=1)
        '''
        x_7: (Batch * speaker) * (Freq + Embedding) * Time
        '''
        #**************Linear*************
        #(Batch * speaker) * (Freq + Embedding) * Time =>(Batch * speaker) * Time * Linear_dim
        x_8 = self.relu(self.Linear(x_7.transpose(1, 2)))
        if split_seg > 0: #batchsize * self.output_speaker, Freq, Time
            num_seg = Time//split_seg
            x_8 = x_8.reshape(batchsize*self.output_speaker, num_seg, split_seg, -1).reshape(batchsize*self.output_speaker*num_seg, split_seg, -1)
            lens = [split_seg for i in range(batchsize*self.output_speaker*num_seg)] 
        #Shared_BLSTMP_1 (Batch * speaker) * Time * Linear_dim => (Batch * speaker) * Time * (Linear_Shared_layer1_dim * 2)
        else:
            lens = [n for n in nframes for i in range(self.output_speaker)] 
        x_9 = self.Shared_BLSTMP_1(x_8, lens)
        #Shared_BLSTMP_2 (Batch * speaker) * Time * (Linear_Shared_layer1_dim * 2) => (Batch * speaker) * Time * (Linear_Shared_layer2_dim * 2)
        x_10 = self.Shared_BLSTMP_2(x_9, lens)

        #Combine-Speaker: (Batch * speaker) * Time * (Linear_Shared_layer2_dim * 2) => Batch * Time * (speaker * Linear_Shared_layer2_dim * 2)
        if split_seg > 0:
            x_11 = x_10.reshape(batchsize, self.output_speaker, num_seg, split_seg, -1).permute(0, 2, 3, 1, 4).reshape(batchsize*num_seg, split_seg, -1)
            lens = [split_seg for i in range(batchsize*num_seg)] 
        else:
            x_11 = x_10.reshape(batchsize, self.output_speaker, Time, -1).transpose(1, 2).reshape(batchsize, Time, -1)
            lens = nframes
        '''
        batchsize * Time * (1stspeaker 2ndspeaker 3rdspeaker 4thspeaker)
        '''

        #BLSTM: Batch * Time * (speaker * Linear_Shared_layer2_dim * 2) => Batch * Time * (BLSTM_Projection_dim * 2)
        x_12 = self.BLSTMP(x_11, lens)

        if split_seg > 0:
            x_12 = x_12.reshape(batchsize, num_seg, split_seg, -1).reshape(batchsize, num_seg*split_seg, -1)
        #Dimension reduction: remove the padding frames; Batch * Time * (BLSTM_Projection_dim * 2) => (Batch * Time) * (BLSTM_Projection_dim * 2)
        lens = [k for i, m in enumerate(nframes) for k in range(i * Time, m + i * Time)]
        x_13 = x_12.reshape(batchsize * Time, -1)[lens, :]
        '''
        1stBatch(1st sentence) length1 * (BLSTM_size * 2)
        2ndBatch(2nd sentence) length2 * (BLSTM_size * 2)
        ...
        '''
        out = []
        for i in self.FC:
            out.append(self.FC[i](x_13))
        if return_embedding:
            return out, embedding
        else:
            return out




class XVEC_MA_MSE_NSD(nn.Module):
    def __init__(self, configs):
        super(XVEC_MA_MSE_NSD, self).__init__()
        self.ResNet = ResNet101(feat_dim=64, embed_dim=256)
        self.SE_MA_MSE_NSD = SE_MA_MSE_NSD(configs)


    def forward(self, x, x_fbank64, mask, nframes, split_seg=-1, return_embedding=False, optimizing_resnet=True):
        '''
        x: Batch * Freq * Time
        x_fbank64: Batch * Freq_64 * Time
        mask : Batch * speaker * Time
        Batch * speaker(4) * Embedding
        nframe: descend order
        split: split long sequence to shorter segments to accelerate BLSTM training
                Time % split_seg == 0
        '''
        batchsize, Freq, Time = x_fbank64.shape
        num_speaker = mask.shape[1]
        #x_1: [batchsize, num_speaker, Time, Fea]
        x_1 = x_fbank64.repeat(1, num_speaker, 1).reshape(batchsize, num_speaker, Freq, Time).transpose(2, 3)

        #x_2: [batchsize, num_speaker, Fea, Time]
        x_2 = (x_1 * mask[..., None]).transpose(2, 3).reshape(batchsize*num_speaker, Freq, Time)

        if optimizing_resnet:
            xvectors = self.ResNet(x_2).reshape(batchsize, num_speaker, -1)
        else:
            with torch.no_grad():
                xvectors = self.ResNet(x_2).reshape(batchsize, num_speaker, -1)
        out = self.SE_MA_MSE_NSD(x, xvectors, mask, nframes, split_seg=split_seg, return_embedding=return_embedding)
        if return_embedding:
            return out, xvectors
        else:
            return out

class XVEC_MA_MSE_NSD2(nn.Module):
    def __init__(self, configs):
        super(XVEC_MA_MSE_NSD2, self).__init__()
        self.ResNet = ResNet101(feat_dim=64, embed_dim=256)
        self.SE_MA_MSE_NSD = SE_MA_MSE_NSD(configs)


    def forward(self, x_fbank64, mask, nframes, split_seg=-1, return_embedding=False, optimizing_resnet=True):
        '''
        x: Batch * Freq * Time
        x_fbank64: Batch * Freq_64 * Time
        mask : Batch * speaker * Time
        Batch * speaker(4) * Embedding
        nframe: descend order
        split: split long sequence to shorter segments to accelerate BLSTM training
                Time % split_seg == 0
        '''
        batchsize, Freq, Time = x_fbank64.shape
        num_speaker = mask.shape[1]
        #x_1: [batchsize, num_speaker, Time, Fea]
        x_1 = x_fbank64.repeat(1, num_speaker, 1).reshape(batchsize, num_speaker, Freq, Time).transpose(2, 3)

        #x_2: [batchsize, num_speaker, Fea, Time]
        x_2 = (x_1 * mask[..., None]).transpose(2, 3).reshape(batchsize*num_speaker, Freq, Time)
        if optimizing_resnet:
            xvectors = self.ResNet(x_2).reshape(batchsize, num_speaker, -1)
        else:
            with torch.no_grad():
                xvectors = self.ResNet(x_2).reshape(batchsize, num_speaker, -1)
        out = self.SE_MA_MSE_NSD(x_fbank64, xvectors, mask, nframes, split_seg=split_seg, return_embedding=return_embedding)
        if return_embedding:
            return out, xvectors
        else:
            return out
