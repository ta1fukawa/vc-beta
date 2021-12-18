import torch

class NoActivation(torch.nn.Module):
    def forward(self, x):
        return x

def get_activation(name, **kwargs):
    if name == 'linear':
        return NoActivation()
    elif name == 'relu':
        return torch.nn.ReLU(**kwargs)
    elif name == 'sigmoid':
        return torch.nn.Sigmoid(**kwargs)
    elif name == 'tanh':
        return torch.nn.Tanh(**kwargs)
    elif name == 'softmax':
        return torch.nn.Softmax(**kwargs)
    elif name == 'linear':
        return torch.nn.Linear(**kwargs)
    else:
        raise ValueError(f'Unknown activation function: {name}')

class Linear(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(Linear, self).__init__()

        self.linear = torch.nn.Linear(in_dim, out_dim, bias)
        torch.nn.init.xavier_uniform_(self.linear.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear(x)

class Conv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, w_init_gain='linear'):
        super(Conv1d, self).__init__()

        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.conv(x)

class Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, w_init_gain='linear'):
        super(Conv2d, self).__init__()

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.conv(x)

class AutoVC(torch.nn.Module):
    def __init__(self, config):
        super(AutoVC, self).__init__()

        self.content_encoder = ContentEncoder(**config['content_encoder'])
        self.prenet          = PreNet(**config['prenet'])
        self.decoder         = Decoder(**config['decoder'])
        self.postnet         = PostNet(**config['postnet'])

        for p in self.speaker_encoder.parameters():
            p.requires_grad_(False)

    def forward(self, src_mel, src_emb, tgt_emb=None):
        if tgt_emb is None:
            tgt_emb = src_emb

        src_cnt = self.content_encoder(src_mel, src_emb)
        mid = torch.cat([
            torch.cat([code.unsqueeze(1).expand(-1, int(src_mel.size(1) / len(src_cnt)), -1) for code in src_cnt], dim=1),
            tgt_emb.unsqueeze(1).expand(-1, src_mel.size(1), -1)
        ], dim=-1)
        rec_mel = self.decoder(self.prenet(mid))
        pst_mel = src_cnt + self.postnet(rec_mel.transpose(2, 1)).transpose(2, 1)
        pst_cnt = torch.cat(self.content_encoder(pst_mel, src_emb), dim=-1)

        return src_cnt, rec_mel, pst_mel, pst_cnt

class ContentEncoder(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_neck, dim_emb, n_layers, n_lstm_layers, kernel_size, stride, dilation, activation='relu'):
        super(ContentEncoder, self).__init__()

        self.conv_layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    Conv1d(
                        dim_in + dim_emb if i == 0 else dim_hidden,
                        dim_hidden,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding='same',
                        dilation=dilation,
                        w_init_gain=activation
                    ),
                    torch.nn.BatchNorm1d(dim_hidden),
                    get_activation(activation),
                ) for i in range(n_layers)
            ]
        )

        self.lstm = torch.nn.LSTM(dim_hidden, dim_neck, n_lstm_layers, batch_first=True, bidirectional=True)

    def forward(self, mel, emb):
        mel = mel.squeeze(1).transpose(2, 1)
        emb = emb.squeeze(-1).expand(-1, -1, mel.size(1))
        x = torch.cat([mel, emb], dim=-1)

        for conv in self.conv_layers:
            x = conv(x)
        
        x, _ = self.lstm(x)
        cnt  = torch.cat([
            x[:, self.skip_len - 1::self.skip_len, :self.dim_neck],
            x[:, :-self.skip_len + 1:self.skip_len, self.dim_neck:]
        ], dim=-1).transpose(0, 1)

        return cnt

class PreNet(torch.nn.Module):
    def __init__(self, dim_neck, dim_emb, dim_out, n_layers):
        super(PreNet, self).__init__()

        self.lstm = torch.nn.LSTM(dim_neck * 2 + dim_emb, dim_out, n_layers, batch_first=True)
    
    def forward(self, mid):
        mid, _ = self.lstm(mid)
        mid    = mid.transpose(2, 1)

        return mid

class Decoder(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, n_layers, n_lstm_layers, kernel_size, stride, dilation, activation='relu'):
        super(Decoder, self).__init__()

        self.conv_layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    Conv1d(
                        dim_in,
                        dim_in,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding='same',
                        dilation=dilation,
                        w_init_gain=activation
                    ),
                    torch.nn.BatchNorm1d(dim_in),
                    get_activation(activation),
                ) for _ in range(n_layers)
            ]
        )

        self.lstm   = torch.nn.LSTM(dim_in, dim_hidden, n_lstm_layers, batch_first=True)
        self.linear = torch.nn.Linear(dim_hidden, dim_out)

    def forward(self, mid):
        x = mid

        for conv in self.conv_layers:
            x = conv(x)
        x = x.transpose(2, 1)

        x, _ = self.lstm(x)
        mel  = self.linear(x)

        return mel

class PostNet(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, n_layers, kernel_size, stride, dilation, activation='tanh'):
        super(PostNet, self).__init__()

        self.conv_layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    Conv1d(
                        dim_in if i == 0 else dim_hidden,
                        dim_hidden if i != n_layers - 1 else dim_out,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding='same',
                        dilation=dilation,
                        w_init_gain=activation if i != n_layers - 1 else 'linear'
                    ),
                    torch.nn.BatchNorm1d(dim_in),
                    get_activation(activation),
                ) for i in range(n_layers)
            ]
        )
    
    def forward(self, mel):
        for conv in self.conv_layers:
            mel = conv(mel)

        return mel