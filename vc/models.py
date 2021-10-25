import warnings

import numpy as np
import torch


def test():
    warnings.filterwarnings('ignore')

    source_speaker_spectrogram = torch.from_numpy(np.random.rand(1, 120, 512)).float()
    print('source_speaker_spectrogram:', source_speaker_spectrogram.shape)
    source_content_spectrogram = torch.from_numpy(np.random.rand(1, 120, 512)).float()
    print('source_content_spectrogram:', source_content_spectrogram.shape)

    print()

    speaker_classfier = SpeakerClassfier()
    speaker_estimate = speaker_classfier(source_speaker_spectrogram)
    print('speaker_estimate:', speaker_estimate.shape)
    speaker_embedding = speaker_classfier.embed(source_speaker_spectrogram)
    print('speaker_embedding:', speaker_embedding.shape)

    print()

    content_encoder_conv1d = ContentEncoderConv1d()
    content_encoder_conv1d_with_speaker = ContentEncoderConv1d(512)
    decoder_conv1d = DecoderConv1d()

    content_embedding_conv1d = content_encoder_conv1d(source_content_spectrogram)
    print('content_embedding_conv1d:', content_embedding_conv1d.shape)
    target_spectrogram_conv1d = decoder_conv1d(speaker_embedding, content_embedding_conv1d)
    print('target_spectrogram_conv1d:', target_spectrogram_conv1d.shape)
    
    content_embedding_conv1d_with_speaker = content_encoder_conv1d_with_speaker(source_content_spectrogram, speaker_embedding)
    print('content_embedding_conv1d_with_speaker:', content_embedding_conv1d_with_speaker.shape)
    target_spectrogram_conv1d_with_speaker = decoder_conv1d(speaker_embedding, content_embedding_conv1d_with_speaker)
    print('target_spectrogram_conv1d_with_speaker:', target_spectrogram_conv1d_with_speaker.shape)

    print()

    content_encoder_conv2d = ContentEncoderConv2d()
    content_encoder_conv2d_with_speaker = ContentEncoderConv2d(512)
    decoder_conv2d = DecoderConv2d()

    content_embedding_conv2d = content_encoder_conv2d(source_content_spectrogram)
    print('content_embedding_conv2d:', content_embedding_conv2d.shape)
    target_spectrogram_conv2d = decoder_conv2d(speaker_embedding, content_embedding_conv2d, content_encoder_conv2d.pool_indices)
    print('target_spectrogram_conv2d:', target_spectrogram_conv2d.shape)

    content_embedding_conv2d_with_speaker = content_encoder_conv2d_with_speaker(source_content_spectrogram, speaker_embedding)
    print('content_embedding_conv2d_with_speaker:', content_embedding_conv2d_with_speaker.shape)
    target_spectrogram_conv2d_with_speaker = decoder_conv2d(speaker_embedding, content_embedding_conv2d_with_speaker, content_encoder_conv2d_with_speaker.pool_indices)
    print('target_spectrogram_conv2d_with_speaker:', target_spectrogram_conv2d_with_speaker.shape)

    print()

    content_encoder_lstm = ContentEncoderLSTM(100, 5)
    content_encoder_lstm_with_speaker = ContentEncoderLSTM(100, 5, 512)
    decoder_lstm = DecoderLSTM(100, 5)
    
    content_embedding_lstm = content_encoder_lstm(source_content_spectrogram)
    print('content_embedding_lstm:', content_embedding_lstm.shape)
    target_spectrogram_lstm = decoder_lstm(speaker_embedding, content_embedding_lstm)
    print('target_spectrogram_lstm:', target_spectrogram_lstm.shape)
    
    content_embedding_lstm_with_speaker = content_encoder_lstm_with_speaker(source_content_spectrogram, speaker_embedding)
    print('content_embedding_lstm_with_speaker:', content_embedding_lstm_with_speaker.shape)
    target_spectrogram_lstm_with_speaker = decoder_lstm(speaker_embedding, content_embedding_lstm_with_speaker)
    print('target_spectrogram_lstm_with_speaker:', target_spectrogram_lstm_with_speaker.shape)

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
    def __init__(self, speaker_embed_dim=0):
        super(ContentEncoderConv1d, self).__init__()
        dims = [512, 256, 128, 64]
        
        self.first_conv = torch.nn.Conv1d(512 + speaker_embed_dim, dims[0], kernel_size=5, dilation=1, padding='same')

        self.layers = torch.nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(torch.nn.Conv1d(dims[i - 1], dims[i], kernel_size=5, dilation=1, padding='same'))
            self.layers.append(torch.nn.Conv1d(dims[i],     dims[i], kernel_size=5, dilation=1, padding='same'))
            self.layers.append(torch.nn.Dropout(p=0.2))

        self.last_conv = torch.nn.Conv1d(dims[-1], 64, kernel_size=5, dilation=1, padding='same')

    def forward(self, content, speaker_embed=None):
        if speaker_embed is not None:
            speaker_embed = torch.unsqueeze(speaker_embed, 1)
            speaker_embed = speaker_embed.expand(-1, content.shape[1], -1)
            x = torch.cat((content, speaker_embed), dim=2)
        else:
            x = content
        x = torch.transpose(x, 1, 2)

        x = self.first_conv(x)

        for layer in self.layers:
            if type(layer) == torch.nn.Conv1d:
                x = torch.nn.functional.relu(layer(x))
            else:
                x = layer(x)

        x = self.last_conv(x)

        return x

class DecoderConv1d(torch.nn.Module):
    def __init__(self):
        super(DecoderConv1d, self).__init__()
        dims = [64, 128, 256, 512]

        self.first_conv = torch.nn.Conv1d(64 + 512, dims[0], kernel_size=5, dilation=1, padding='same')

        self.layers = torch.nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(torch.nn.Dropout(p=0.2))
            self.layers.append(torch.nn.Conv1d(dims[i], dims[i],     kernel_size=5, dilation=1, padding='same'))
            self.layers.append(torch.nn.Conv1d(dims[i], dims[i + 1], kernel_size=5, dilation=1, padding='same'))

        self.last_conv = torch.nn.Conv1d(dims[-1], 512, kernel_size=5, dilation=1, padding='same')

    def forward(self, speaker_embed, content_embed):
        speaker_embed = torch.unsqueeze(speaker_embed, -1)
        speaker_embed = speaker_embed.expand(-1, -1, content_embed.shape[-1])
        x = torch.cat((content_embed, speaker_embed), dim=1)
        x = self.first_conv(x)

        for layer in self.layers:
            if type(layer) == torch.nn.Conv1d:
                x = torch.nn.functional.relu(layer(x))
            else:
                x = layer(x)
        
        x = self.last_conv(x)
        x = torch.transpose(x, 1, 2)
        return x

class ContentEncoderConv2d(torch.nn.Module):
    def __init__(self, speaker_embed_dim=0):
        super(ContentEncoderConv2d, self).__init__()
        dims = [1, 32, 64, 128]
        
        self.first_conv = torch.nn.Conv2d(1 + speaker_embed_dim, dims[0], kernel_size=(5, 5), dilation=(1, 1), padding='same')

        self.layers = torch.nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(torch.nn.Conv2d(dims[i - 1], dims[i], kernel_size=(5, 5), dilation=(1, 1), padding='same'))
            self.layers.append(torch.nn.Conv2d(dims[i],     dims[i], kernel_size=(5, 5), dilation=(1, 1), padding='same'))
            self.layers.append(torch.nn.Dropout2d(p=0.2))
            # self.layers.append(torch.nn.MaxPool2d(kernel_size=(1, 4), return_indices=True))

        self.last_conv = torch.nn.Conv2d(dims[-1], 128, kernel_size=(5, 5), dilation=(1, 1), padding='same')

    def forward(self, content, speaker_embed=None):
        content = torch.unsqueeze(content, 1)
        if speaker_embed is not None:
            speaker_embed = torch.unsqueeze(speaker_embed, 2)
            speaker_embed = torch.unsqueeze(speaker_embed, 3)
            speaker_embed = speaker_embed.expand(-1, -1, content.shape[2], content.shape[3])
            x = torch.cat((content, speaker_embed), dim=1)
            x = self.first_conv(x)
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

        x = self.last_conv(x)
        
        return x

class DecoderConv2d(torch.nn.Module):
    def __init__(self):
        super(DecoderConv2d, self).__init__()
        dims = [128, 64, 32, 1]

        self.first_conv = torch.nn.Conv2d(128 + 512, dims[0], kernel_size=(5, 5), dilation=(1, 1), padding='same')

        self.layers = torch.nn.ModuleList()
        for i in range(len(dims) - 1):
            # self.layers.append(torch.nn.MaxUnpool2d(kernel_size=(1, 4)))
            self.layers.append(torch.nn.Dropout2d(p=0.2))
            self.layers.append(torch.nn.Conv2d(dims[i], dims[i],     kernel_size=(5, 5), dilation=(1, 1), padding='same'))
            self.layers.append(torch.nn.Conv2d(dims[i], dims[i + 1], kernel_size=(5, 5), dilation=(1, 1), padding='same'))

        self.last_conv = torch.nn.Conv2d(dims[-1], 1, kernel_size=(5, 5), dilation=(1, 1), padding='same')

    def forward(self, speaker_embed, content_embed, pool_indices):
        speaker_embed = torch.unsqueeze(speaker_embed, 2)
        speaker_embed = torch.unsqueeze(speaker_embed, 3)
        speaker_embed = speaker_embed.expand(-1, -1, content_embed.shape[2], content_embed.shape[3])
        x = torch.cat((content_embed, speaker_embed), dim=1)
        x = self.first_conv(x)

        pool_indices_i = -1
        for layer in self.layers:
            if type(layer) == torch.nn.Conv2d:
                x = torch.nn.functional.relu(layer(x))
            elif type(layer) == torch.nn.MaxUnpool2d:
                x = layer(x, pool_indices[pool_indices_i])
                pool_indices_i -= 1
            else:
                x = layer(x)

        x = self.last_conv(x)
        x = torch.squeeze(x, 1)
        return x

class ContentEncoderLSTM(torch.nn.Module):
    def __init__(self, dim_neck, skip_len, speaker_embed_dim=0):
        super(ContentEncoderLSTM, self).__init__()
        self.dim_neck = dim_neck
        self.skip_len = skip_len
        dims = [512, 512, 512, 512]

        self.first_conv = torch.nn.Conv1d(512 + speaker_embed_dim, dims[0], kernel_size=5, dilation=1, padding='same')

        self.layers = torch.nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(torch.nn.Conv1d(dims[i - 1], dims[i], kernel_size=5, dilation=1, padding='same'))
            self.layers.append(torch.nn.Dropout(p=0.2))

        self.lstm = torch.nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, content, speaker_embed=None):
        if speaker_embed is not None:
            speaker_embed = torch.unsqueeze(speaker_embed, 1)
            speaker_embed = speaker_embed.expand(-1, content.shape[1], -1)
            x = torch.cat((content, speaker_embed), dim=2)
        else:
            x = content
        x = torch.transpose(x, 1, 2)

        x = self.first_conv(x)

        for layer in self.layers:
            if type(layer) == torch.nn.Conv1d:
                x = torch.nn.functional.relu(layer(x))
            else:
                x = layer(x)

        x = torch.transpose(x, 1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = torch.cat([x[:, self.skip_len - 1::self.skip_len, :self.dim_neck], x[:, :-self.skip_len + 1:self.skip_len, self.dim_neck:]], dim=-1)
        x = torch.transpose(x, 1, 2)

        return x

class DecoderLSTM(torch.nn.Module):
    def __init__(self, dim_neck, skip_len):
        super(DecoderLSTM, self).__init__()
        self.skip_len = skip_len
        dims = [512, 512, 512, 512]

        self.lstm1 = torch.nn.LSTM(dim_neck * 2 + 512, 512, 1, batch_first=True)

        self.layers = torch.nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(torch.nn.Dropout(p=0.2))
            self.layers.append(torch.nn.Conv1d(dims[i], dims[i],     kernel_size=5, dilation=1, padding='same'))
            self.layers.append(torch.nn.Conv1d(dims[i], dims[i + 1], kernel_size=5, dilation=1, padding='same'))
            
        self.lstm2 = torch.nn.LSTM(512, 1024, 2, batch_first=True)
        self.linear = torch.nn.Linear(1024, 512)

    def forward(self, speaker_embed, content_embed):
        content_embed = torch.repeat_interleave(content_embed, self.skip_len, dim=2)

        speaker_embed = torch.unsqueeze(speaker_embed, -1)
        speaker_embed = speaker_embed.expand(-1, -1, content_embed.shape[-1])
        x = torch.cat((content_embed, speaker_embed), dim=1)

        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm1(x)
        x = torch.transpose(x, 1, 2)

        for layer in self.layers:
            if type(layer) == torch.nn.Conv1d:
                x = torch.nn.functional.relu(layer(x))
            else:
                x = layer(x)

        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm2(x)
        x = self.linear(x)
        
        return x
