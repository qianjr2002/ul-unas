# refer to https://github.com/Xiaobin-Rong/ul-unas/issues/2#issue-3002098373

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class TRA(nn.Module):
    """Temporal Recurrent Attention"""

    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels * 2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """x"""
        zt = torch.mean(x.pow(2), dim=-1)
        at = self.att_gru(zt.transpose(1, 2))[0]
        at = self.att_fc(at)
        At = self.att_act(at).transpose(1, 2).unsqueeze(-1)

        return At


class cTFA(nn.Module):
    def __init__(self, in_channels, r=1):
        super(cTFA, self).__init__()

        self.tra = TRA(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels * r, kernel_size=(3, 1), padding=(1, 0))
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels * r, in_channels, kernel_size=(3, 1), padding=(1, 0))
        self.sigmoid_fdta = nn.Sigmoid()

    def forward(self, x):
        out_fita = self.tra(x)

        out_fdta = self.conv1(out_fita)
        out_fdta = self.prelu(out_fdta)
        out_fdta = self.conv2(out_fdta)
        out_fdta = self.sigmoid_fdta(out_fdta)

        out = out_fita * out_fdta

        return out * x


class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs - erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs - erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        erb_f = 21.4 * np.log10(0.00437 * freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        freq_hz = (10 ** (erb_f / 21.4) - 1) / 0.00437
        return freq_hz

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = erb_subband_1 / nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points) / fs * nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
                                          / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2 - 2):
            erb_filters[i + 1, bins[i]:bins[i + 1]] = (np.arange(bins[i], bins[i + 1]) - bins[i] + 1e-12) \
                                                      / (bins[i + 1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i + 1]:bins[i + 2]] = (bins[i + 2] - np.arange(bins[i + 1], bins[i + 2]) + 1e-12) \
                                                          / (bins[i + 2] - bins[i + 1] + 1e-12)

        erb_filters[-1, bins[-2]:bins[-1] + 1] = 1 - erb_filters[-2, bins[-2]:bins[-1] + 1]

        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))

    def bm(self, x):
        """x"""
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1)

    def bs(self, x_erb):
        """x_erb"""
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)

    
class APReLU(nn.Module):
    def __init__(self, num_channels, num_frequencies=None):  # ignore F
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_channels))  # [C]
        self.beta = nn.Parameter(torch.zeros(num_channels))  # [C]
        self.alpha = nn.Parameter(torch.full((num_channels,), 0.25))  # [C]

    def forward(self, x):  # [B, C, T, F]
        x_affine = self.gamma.view(1, -1, 1, 1) * x + self.beta.view(1, -1, 1, 1)
        x_prelu = F.relu(x) + self.alpha.view(1, -1, 1, 1) * F.relu(-x)
        return x_affine + x_prelu



class XConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, nF=161):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = APReLU(out_channels, nF)
        self.tra = cTFA(out_channels)

    def forward(self, x):
        return self.tra(self.act(self.bn(self.conv(x))))


class XDWS_block(nn.Module):
    """Group Temporal Convolution"""

    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, groups, nF, use_deconv=False):
        super().__init__()
        self.use_deconv = use_deconv
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d

        self.point_conv1 = conv_module(in_channels, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = APReLU(hidden_channels, nF)

        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                      stride=stride, padding=padding, groups=groups)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = APReLU(hidden_channels, nF)

        self.tra = cTFA(hidden_channels)

    def forward(self, x):
        """x"""
        x = self.point_act(self.point_bn1(self.point_conv1(x)))
        x = self.depth_act(self.depth_bn(self.depth_conv(x)))
        x = self.tra(x)
        return x


class XMB_block(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, groups, nF, use_deconv=False):
        super().__init__()
        self.use_deconv = use_deconv
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d

        self.point_conv1 = conv_module(in_channels, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = APReLU(hidden_channels, nF)

        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                      stride=stride, padding=padding, groups=groups)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        if stride[1] == 2:
            f_size = nF * 2 - 1 if use_deconv else nF // stride[1] + 1
        else:
            f_size = nF
        self.depth_act = APReLU(hidden_channels, f_size)

        self.point_conv2 = conv_module(hidden_channels, hidden_channels, 1)
        self.point_bn2 = nn.BatchNorm2d(hidden_channels)

        self.tra = cTFA(hidden_channels)

    def forward(self, x):
        """x"""
        # x_ref = x
        x = self.point_act(self.point_bn1(self.point_conv1(x)))
        x = self.depth_act(self.depth_bn(self.depth_conv(x)))
        x = self.point_bn2(self.point_conv2(x))
        x = self.tra(x)
        return x


class GRNN(nn.Module):
    """Grouped RNN"""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size // 2, hidden_size // 2, num_layers, batch_first=batch_first,
                           bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size // 2, hidden_size // 2, num_layers, batch_first=batch_first,
                           bidirectional=bidirectional)

    def forward(self, x, h=None):
        """
        x
        h
        """
        if h == None:
            if self.bidirectional:
                h = torch.zeros(self.num_layers * 2, x.shape[0], self.hidden_size, device=x.device)
            else:
                h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return y, h


class DPGRNN(nn.Module):
    def __init__(self, input_size, width, hidden_size, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, input_size)
        self.intra_ln = nn.LayerNorm((width, input_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size // 2, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size // 2, input_size)
        self.inter_ln = nn.LayerNorm(((width, input_size)), eps=1e-8)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        intra_x = self.intra_rnn(intra_x)[0]
        intra_x = self.intra_fc(intra_x)
        intra_x = intra_x.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        intra_x = self.intra_ln(intra_x)

        intra_out = torch.add(x, intra_x)

        ## Inter RNN
        x = intra_out.permute(0, 2, 1, 3)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        inter_x = self.inter_rnn(inter_x)[0]
        inter_x = self.inter_fc(inter_x)
        inter_x = inter_x.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        inter_x = inter_x.permute(0, 2, 1, 3)
        inter_x = self.inter_ln(inter_x)
        inter_out = torch.add(intra_out, inter_x)

        dual_out = inter_out.permute(0, 3, 1, 2)

        return dual_out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            XConvBlock(1, 12, (3, 3), stride=(1, 2), padding=(0, 1), use_deconv=False, nF=81),
            XMB_block(12, 24, (2, 3), stride=(1, 2), padding=(0, 1), groups=2, use_deconv=False, nF=81),
            XDWS_block(24, 24, (2, 3), stride=(1, 1), padding=(0, 1), groups=2, use_deconv=False, nF=41),
            XMB_block(24, 32, (1, 5), stride=(1, 1), padding=(0, 2), groups=2, use_deconv=False, nF=41),
            XDWS_block(32, 16, (1, 5), stride=(1, 1), padding=(0, 2), groups=2, use_deconv=False, nF=41)
        ])

    def forward(self, x):
        en_outs = []
        for i in range(len(self.en_convs)):
            x = self.en_convs[i](x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList([
            XDWS_block(16, 32, (1, 5), stride=(1, 1), padding=(0, 2), groups=2, use_deconv=True, nF=41),
            XMB_block(32, 24, (1, 5), stride=(1, 1), padding=(0, 2), groups=2, use_deconv=True, nF=41),
            XDWS_block(24, 24, (2, 3), stride=(1, 1), padding=(0, 1), groups=2, use_deconv=True, nF=41),
            XMB_block(24, 12, (2, 3), stride=(1, 2), padding=(0, 1), groups=2, use_deconv=True, nF=41),
            XConvBlock(12, 1, (3, 3), stride=(1, 2), padding=(0, 1), use_deconv=True, nF=161),
        ])

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            x = self.de_convs[i](x + en_outs[N_layers - 1 - i])
        return x


class UL_UNAS(nn.Module):
    def __init__(self):
        super().__init__()
        self.erb = ERB(81, 80)
        self.encoder = Encoder()
        self.dpgrnn1 = DPGRNN(16, 41, 32)
        self.dpgrnn2 = DPGRNN(16, 41, 32)

        self.decoder = Decoder()

    def forward(self, spec):
        spec = spec.permute(0, 1, 3, 2)

        feat = self.erb.bm(spec)

        feat, en_outs = self.encoder(feat)

        feat = self.dpgrnn1(feat)
        feat = self.dpgrnn2(feat)

        m_feat = self.decoder(feat, en_outs)
        m_feat = self.erb.bs(m_feat)
        m_feat = m_feat.permute(0, 1, 3, 2)

        return m_feat


if __name__ == "__main__":
    
    model = UL_UNAS()
    noisy = torch.rand(1, 16000)

    spec = torch.stft(
        noisy,
        512,
        256,
        512,
        window=torch.hann_window(512).pow(0.5),
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    magnitude = torch.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2)
    phase = torch.atan2(spec[..., 1], spec[..., 0])

    print(f'magnitude.shape{magnitude.shape}')  # magnitude.shape: torch.Size([1, 257, 63])

    num_freq_bins = magnitude.shape[-1]  # 257
    num_frames = magnitude.shape[-2]  # 63


    enhanced_magnitude = model(magnitude.unsqueeze(0))
    print(f'magnitude shape: {magnitude.shape}') # magnitude shape: [1, 257, 63]
    print(f'Enhanced magnitude shape: {enhanced_magnitude.shape}') # Enhanced magnitude shape:[1, 1, 257, 63]

    enhanced_magnitude = enhanced_magnitude.squeeze(1)  # [1, 257, 63]

    real = enhanced_magnitude * torch.cos(phase)
    imag = enhanced_magnitude * torch.sin(phase)

    complex_spec = torch.stack([real, imag], dim=-1)  # [1, 257, 63, 2]
    est = torch.view_as_complex(complex_spec)  # [1, 257, 63]

    est_wav = torch.istft(
        est,
        n_fft=512,
        hop_length=256,
        win_length=512,
        window=torch.hann_window(512).pow(0.5),
        return_complex=False,
    )
    print(f'est_wav shape: {est_wav.shape}')  # est_wav shape: [1, 15872]

    input_shape = (1, num_frames, num_freq_bins)  # (1, 257, 63)
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, input_shape,
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             verbose=True)
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
    # Computational complexity:       85.17 MMac
    # Number of parameters:           164.63 k
