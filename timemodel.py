import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # print(x.shape,'-----------2-------------')
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
    
class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
    
def FFT_for_Period(x, k=2):
    # [B, T, C]
    # xf = torch.fft.rfft(x, dim=1)
    xf = torch.tensor(np.fft.fft(x.detach().cpu().numpy(), axis=1))
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self):
        super(TimesBlock, self).__init__()
        self.seq_len = 365
        self.pred_len = 1
        self.k = 5
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(16, 16,
                               num_kernels=3),
            nn.GELU(),
            Inception_Block_V1(16, 16,
                               num_kernels=3)
        )

    def forward(self, x, device):
        B, T, N = x.size()
        # print(x.size(),'---------2-------')
        period_list, period_weight = FFT_for_Period(x, self.k)
        # print(period_list.shape,period_list)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # print(self.seq_len + self.pred_len,period,'-----------5----------')
            # padding
            if self.seq_len % period != 0:
            # if (self.seq_len + self.pred_len) % period != 0:
                
                length = ((self.seq_len // period) + 1) * period
                # length = (((self.seq_len + self.pred_len) // period) + 1) * period
                # print(length,period,'--------6.1------------')
                padding = torch.zeros([x.shape[0], (length - (self.seq_len)), x.shape[2]]).to(x.device)
                # padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                # print(x.shape,padding.shape,'-------------3-----------')
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len)
                # length = (self.seq_len + self.pred_len)
                # print(length,period,'--------6.2------------')
                out = x
            # reshape
            # print(out.shape,B, length // period, period,N,'---------4----------')
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len, :])
            # res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight.to(device), -1)
        # residual connection
        res = res + x
        return res

class TimeModel(nn.Module):
    def __init__(self):
        super(TimeModel, self).__init__()
        enc_in = 1
        d_pre = 365
        d_model = 16
        pred_len = 1
        self.enc_embedding = DataEmbedding(enc_in, d_model)
        # self.predict_linear = nn.Linear(d_pre, d_pre)
        # self.projection = nn.Linear(d_model, enc_in, bias=True)
        # self.model = nn.ModuleList([TimesBlock() for _ in range(2)])
        self.model =TimesBlock()
        self.layer_norm = nn.LayerNorm(d_model)
    def forward(self, inputs,device):
        # print(inputs.size())
        B,T,C = inputs.size()
        Ns = []
        for i in range(C):
            temp = torch.unsqueeze(inputs[:,:,i], 2)
            # print(temp.shape,'---------3----------')
            enc_out = self.enc_embedding(temp) 
            # enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
            # for i in range(2):
            #     enc_out = self.model[i](enc_out,device)
            #     enc_out = self.layer_norm(enc_out.float())
            # print(enc_out.shape,i,'-------------4--------------')            
            # dec_out = self.projection(enc_out)
            # print(dec_out.shape,i,'-------------5--------------')
            enc_out = self.model(enc_out,device)
            Ns.append(enc_out)
        Ns = torch.dstack(Ns)
        # print(Ns.shape,'---------5--------')
        return Ns
            
        
