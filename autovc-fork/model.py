import torch

class DVector(torch.nn.Module):

    def __init__(self, nlayers=3, ninput=40, nhidden=256, noutput=64):
        super(DVector, self).__init__()
        self.lstm = torch.nn.LSTM(ninput, nhidden, nlayers, batch_first=True)
        self.emb  = torch.nn.Linear(nhidden, noutput)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = self.emb(x[:, -1, :])
        x = x.div(x.norm(p=2, dim=-1, keepdim=True))
        return x

class XVectorConv2D(torch.nn.Module):
    def __init__(self):
        super(XVectorConv2D, self).__init__()
        
        self.conv1a = torch.nn.Conv2d(1,  64, kernel_size=(3, 3), dilation=(1, 1), padding='same')
        self.conv1b = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), dilation=(1, 1), padding='same')
        self.drop1  = torch.nn.Dropout2d(p=0.2)
        self.pool1  = torch.nn.MaxPool2d(kernel_size=(4, 4))
        
        self.conv2a = torch.nn.Conv2d(64,  128, kernel_size=(3, 3), dilation=(1, 1), padding='same')
        self.conv2b = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), dilation=(1, 1), padding='same')
        self.drop2  = torch.nn.Dropout2d(p=0.2)
        self.pool2  = torch.nn.MaxPool2d(kernel_size=(4, 4))
        
        self.conv3a = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), dilation=(1, 1), padding='same')
        self.conv3b = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), dilation=(1, 1), padding='same')
        self.drop3  = torch.nn.Dropout2d(p=0.2)
        self.pool3  = torch.nn.MaxPool2d(kernel_size=(4, 4))
        
        self.conv4  = torch.nn.Conv2d(256, 2048, kernel_size=(3, 3), dilation=(1, 1), padding='same')
        self.line4  = torch.nn.Linear(2048, 512)
        
    def _max_pooling(self, x):
        return x.max(dim=3)[0].max(dim=2)[0]
        
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        
        x = torch.nn.functional.relu(self.conv1a(x))
        x = torch.nn.functional.relu(self.conv1b(x))
        x = self.drop1(x)
        x = self.pool1(x)
        
        x = torch.nn.functional.relu(self.conv2a(x))
        x = torch.nn.functional.relu(self.conv2b(x))
        x = self.drop2(x)
        x = self.pool2(x)
        
        x = torch.nn.functional.relu(self.conv3a(x))
        x = torch.nn.functional.relu(self.conv3b(x))
        x = self.drop3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self._max_pooling(x)
        x = self.line4(x)
        return x

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.line = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(self.line.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.line(x)

class Conv1dNorm(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels,
        kernel_size=1, stride=1, padding=None, dilation=1, bias=True,
        w_init_gain='linear'
    ):
        super(Conv1dNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )

        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class Encoder(torch.nn.Module):
    def __init__(self, dim_neck, skip_len):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.skip_len = skip_len

        self.convs = torch.nn.ModuleList()
        for i in range(3):
            in_channels = 80 + 512 if i == 0 else 512
            conv = torch.nn.Sequential(
                Conv1dNorm(in_channels, 512, kernel_size=5, padding=2, w_init_gain='relu'),
                torch.nn.BatchNorm1d(512),
            )
            self.convs.append(conv)

        self.lstm = torch.nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, src_uttr, src_emb):
        src_uttr = torch.transpose(torch.squeeze(src_uttr, 1), 1, 2)
        src_emb = torch.unsqueeze(src_emb, -1).expand(-1, -1, src_uttr.size(-1))
        x = torch.cat([src_uttr, src_emb], dim=1)

        for conv in self.convs:
            x = torch.nn.functional.relu(conv(x))
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = torch.cat([
            x[:, self.skip_len - 1::self.skip_len, :self.dim_neck],
            x[:, :-self.skip_len + 1:self.skip_len, self.dim_neck:]
        ], dim=-1)
        x = torch.transpose(x, 0, 1)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, dim_neck):
        super(Decoder, self).__init__()
        
        self.lstm1 = torch.nn.LSTM(dim_neck * 2 + 512, 512, 1, batch_first=True)

        self.convs = torch.nn.ModuleList()
        for i in range(3):
            conv = torch.nn.Sequential(
                Conv1dNorm(512, 512, kernel_size=5, padding=2, w_init_gain='relu'),
                torch.nn.BatchNorm1d(512),
            )
            self.convs.append(conv)
        
        self.lstm2 = torch.nn.LSTM(512, 1024, 2, batch_first=True)
        self.line = LinearNorm(1024, 80)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)

        for conv in self.convs:
            x = torch.nn.functional.relu(conv(x))
        x = x.transpose(1, 2)

        tgt_uttr, _ = self.lstm2(x)
        tgt_uttr = self.line(tgt_uttr)
        return tgt_uttr

class Postnet(torch.nn.Module):
    def __init__(self):
        super(Postnet, self).__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(5):
            in_channels = 80 if i == 0 else 512
            out_channels = 80 if i == 4 else 512
            conv = torch.nn.Sequential(
                Conv1dNorm(in_channels, out_channels, kernel_size=5, padding=2, w_init_gain='relu'),
                torch.nn.BatchNorm1d(out_channels),
            )
            self.convs.append(conv)
    
    def forward(self, tgt):
        for conv in self.convs[:-1]:
            tgt = torch.tanh(conv(tgt))
        tgt = self.convs[-1](tgt)
        return tgt

class AutoVC(torch.nn.Module):
    def __init__(self, dim_neck, skip_len):
        super(AutoVC, self).__init__()

        self.encoder = Encoder(dim_neck, skip_len)
        self.decoder = Decoder(dim_neck)
        self.postnet = Postnet()

    def forward(self, src_uttr, src_emb, tgt_emb):
        codes = self.encoder(src_uttr, src_emb)

        tmp = []
        for code in codes:
            tmp.append(torch.unsqueeze(code, 1).expand(-1, src_uttr.shape[1] // len(codes), -1))
        code_exp = torch.cat(tmp, dim=1)

        content = torch.cat((code_exp, torch.unsqueeze(tgt_emb, 1).expand(-1, src_uttr.shape[1], -1)), dim=-1)

        tgt_uttr = self.decoder(content)
        tgt_psnt = tgt_uttr + torch.transpose(self.postnet(torch.transpose(tgt_uttr, 1, 2)), 1, 2)

        tgt_uttr = tgt_uttr.unsqueeze(1)
        tgt_psnt = tgt_psnt.unsqueeze(1)

        return tgt_uttr, tgt_psnt, codes