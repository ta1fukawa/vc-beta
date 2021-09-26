import numpy as np
from numpy.core.numeric import indices
import torch

class SpeakerEncoder(torch.nn.Module):
    def __init__(self):
        super(SpeakerEncoder, self).__init__()

        dims = [1, 64, 128, 256]
        self.layers = torch.nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(torch.nn.Conv2d(dims[i - 1], dims[i], kernel_size=(5, 5), dilation=(1, 1), padding='same'))
            self.layers.append(torch.nn.Conv2d(dims[i],     dims[i], kernel_size=(5, 5), dilation=(1, 1), padding='same'))
            self.layers.append(torch.nn.Dropout2d(p=0.2))
            self.layers.append(torch.nn.MaxPool2d(kernel_size=(1, 4)))

        self.conv = torch.nn.Conv2d(dims[-1], 2048, kernel_size=(5, 5), dilation=(1, 1), padding='same')
        self.line = torch.nn.Linear(2048, 512)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)

        for layer in self.layers:
            if type(layer) == torch.nn.Conv2d:
                x = torch.nn.functional.relu(layer(x))
            else:
                x = layer(x)

        x = self.conv(x)
        x = x.max(dim=3)[0].max(dim=2)[0]
        x = self.line(x)

        return x

class SpeakerClassfier(torch.nn.Module):
    def __init__(self, nclasses=16):
        super(SpeakerClassfier, self).__init__()
        
        self.embed = SpeakerEncoder()
        self.drop1 = torch.nn.Dropout(p=0.2)
        
        self.line2 = torch.nn.Linear(512, 1024)
        self.drop2 = torch.nn.Dropout(p=0.2)
        
        self.line3 = torch.nn.Linear(1024, nclasses)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.embed(x))
        x = self.drop1(x)
        
        x = torch.nn.functional.relu(self.line2(x))
        x = self.drop2(x)
        
        x = torch.nn.functional.log_softmax(self.line3(x), dim=-1)
        
        return x

class ContentEncoderConv1d(torch.nn.Module):
    def __init__(self):
        super(ContentEncoderConv1d, self).__init__()
        dims = [512, 512, 512, 512]

        self.layers = torch.nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(torch.nn.Conv1d(dims[i - 1], dims[i], kernel_size=5, dilation=1, padding='same'))
            self.layers.append(torch.nn.Conv1d(dims[i],     dims[i], kernel_size=5, dilation=1, padding='same'))
            self.layers.append(torch.nn.Dropout(p=0.2))
            self.layers.append(torch.nn.MaxPool1d(kernel_size=4), return_indices=True)

        self.conv = torch.nn.Conv1d(dims[-1], 512, kernel_size=5, dilation=1, padding='same')

    def forward(self, content, speaker_embed=None):
        if speaker_embed is not None:
            speaker_embed = torch.unsqueeze(speaker_embed, -1)
            speaker_embed = speaker_embed.expand(-1, -1, x.shape[-1])
            x = torch.cat((content, speaker_embed), dim=1)
        else:
            x = content

        self.pool_indices = list()
        for layer in self.layers:
            if type(layer) == torch.nn.Conv1d:
                x = torch.nn.functional.relu(layer(x))
            elif type(layer) == torch.nn.MaxPool1d:
                x, indices = layer(x)
                self.pool_indices.append(indices)
            else:
                x = layer(x)

        x = self.conv(x)

        return x

class DecoderConv1d(torch.nn.Module):
    def __init__(self):
        super(DecoderConv1d, self).__init__()
        dims = [512, 512, 512, 512]

        self.conv = torch.nn.Conv1d(512 + 512, dims[0])

        self.layers = torch.nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(torch.nn.MaxUnpool1d(kernel_size=4))
            self.layers.append(torch.nn.Dropout(p=0.2))
            self.layers.append(torch.nn.Conv1d(dims[i], dims[i],     kernel_size=5, dilation=1, padding='same'))
            self.layers.append(torch.nn.Conv1d(dims[i], dims[i + 1], kernel_size=5, dilation=1, padding='same'))

    def forward(self, content_encoder, speaker_embed, content_embed):
        speaker_embed = torch.unsqueeze(speaker_embed, -1)
        speaker_embed = speaker_embed.expand(-1, -1, content_embed.shape[-1])
        x = torch.cat((content_embed, speaker_embed), dim=1)
        x = self.conv(x)

        pool_indices_i = 0
        for layer in self.layers:
            if type(layer) == torch.nn.Conv1d:
                x = torch.nn.functional.relu(layer(x))
            elif type(layer) == torch.nn.MaxPool1d:
                x = layer(x, content_encoder.pool_incices[pool_indices_i])
                pool_indices_i += 1
            else:
                x = layer(x)

class ContentEncoderConv2d(torch.nn.Module):
    def __init__(self):
        super(ContentEncoderConv2d, self).__init__()
        dims = [1, 32, 64, 128]

        self.layers = torch.nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(torch.nn.Conv2d(dims[i - 1], dims[i], kernel_size=(5, 5), dilation=(1, 1), padding='same'))
            self.layers.append(torch.nn.Conv2d(dims[i],     dims[i], kernel_size=(5, 5), dilation=(1, 1), padding='same'))
            self.layers.append(torch.nn.Dropout2d(p=0.2))
            self.layers.append(torch.nn.MaxPool2d(kernel_size=(1, 4)), return_indices=True)

        self.conv = torch.nn.Conv2d(dims[-1], 128, kernel_size=(5, 5), dilation=(1, 1), padding='same')

    def forward(self, content, speaker_embed=None):
        if speaker_embed is not None:
            speaker_embed = torch.unsqueeze(speaker_embed, -1)
            speaker_embed = torch.unsqueeze(speaker_embed, -1)
            speaker_embed = speaker_embed.expand(-1, -1, x.shape[-2], x.shape[-1])
            x = torch.cat((content, speaker_embed), dim=1)
        else:
            x = content

        self.pool_indices = list()
        for layer in self.layers:
            if type(layer) == torch.nn.Conv2d:
                x = torch.nn.functional.relu(layer(x))
            elif type(layer) == torch.nn.MaxPool2d:
                x, indices = layer(x)
                self.pool_indices.append(indices)
            else:
                x = layer(x)

        x = self.conv(x)

        return x

class DecoderConv2d(torch.nn.Module):
    def __init__(self):
        super(DecoderConv2d, self).__init__()
        dims = [128, 64, 32, 1]

        self.conv = torch.nn.Conv1d(128 + 512, dims[0])

        self.layers = torch.nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(torch.nn.MaxUnpool1d(kernel_size=4))
            self.layers.append(torch.nn.Dropout(p=0.2))
            self.layers.append(torch.nn.Conv1d(dims[i], dims[i],     kernel_size=5, dilation=1, padding='same'))
            self.layers.append(torch.nn.Conv1d(dims[i], dims[i + 1], kernel_size=5, dilation=1, padding='same'))

    def forward(self, content_encoder, speaker_embed, content_embed):
        speaker_embed = torch.unsqueeze(speaker_embed, -1)
        speaker_embed = torch.unsqueeze(speaker_embed, -1)
        speaker_embed = speaker_embed.expand(-1, -1, content_embed.shape[-2], content_embed.shape[-1])
        x = torch.cat((content_embed, speaker_embed), dim=1)

        x = self.conv(x)

        pool_indices_i = 0
        for layer in self.layers:
            if type(layer) == torch.nn.Conv2d:
                x = torch.nn.functional.relu(layer(x))
            elif type(layer) == torch.nn.MaxPool2d:
                x = layer(x, content_encoder.pool_incices[pool_indices_i])
                pool_indices_i += 1
            else:
                x = layer(x)

class ContentEncoderLSTM(torch.nn.Module):
    def __init__(self, dim_neck, freq):
        super(ContentEncoderLSTM, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq
        dims = [512, 512, 512, 512]

        self.layers = torch.nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(torch.nn.Conv1d(dims[i - 1], dims[i], kernel_size=5, dilation=1, padding='same'))
            self.layers.append(torch.nn.Dropout(p=0.2))

        self.lstm = torch.nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, content, speaker_embed=None):
        if speaker_embed is not None:
            speaker_embed = torch.unsqueeze(speaker_embed, -1)
            speaker_embed = speaker_embed.expand(-1, -1, x.shape[-1])
            x = torch.cat((content, speaker_embed), dim=1)
        else:
            x = content

        for layer in self.layers:
            if type(layer) == torch.nn.Conv1d:
                x = torch.nn.functional.relu(layer(x))
            else:
                x = layer(x)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward  = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]
        
        codes = []
        for i in range(0, outputs.shape[1], self.freq):
            codes.append(torch.cat((out_forward[:, i + self.freq - 1, :], out_backward[:, i, :]), dim=-1))

        return codes

class DecoderLSTM(torch.nn.Module):
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(DecoderLSTM, self).__init__()
        dims = [512, 512, 512, 512]

        self.lstm1 = torch.nn.LSTM(dim_neck * 2 + dim_emb, dim_pre, 1, batch_first=True, bidirectional=True)

        self.layers = torch.nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(torch.nn.MaxUnpool1d(kernel_size=4))
            self.layers.append(torch.nn.Dropout(p=0.2))
            self.layers.append(torch.nn.Conv1d(dims[i], dims[i],     kernel_size=5, dilation=1, padding='same'))
            self.layers.append(torch.nn.Conv1d(dims[i], dims[i + 1], kernel_size=5, dilation=1, padding='same'))

    def forward(self, content_encoder, speaker_embed, content_embed):
        speaker_embed = torch.unsqueeze(speaker_embed, -1)
        speaker_embed = speaker_embed.expand(-1, -1, content_embed.shape[-1])
        x = torch.cat((content_embed, speaker_embed), dim=1)
        x = self.conv(x)

        pool_indices_i = 0
        for layer in self.layers:
            if type(layer) == torch.nn.Conv1d:
                x = torch.nn.functional.relu(layer(x))
            elif type(layer) == torch.nn.MaxPool1d:
                x = layer(x, content_encoder.pool_incices[pool_indices_i])
                pool_indices_i += 1
            else:
                x = layer(x)

class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()
        
        self.lstm1 = nn.LSTM(dim_neck*2+dim_emb, dim_pre, 1, batch_first=True)
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre, dim_pre, kernel_size=5, stride=1, padding=2, dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        
        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x):
        
        #self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        outputs, _ = self.lstm2(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output   