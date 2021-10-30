import torch

class DVectorModel(torch.nn.Module):
    def __init__(self, num_layers=3, dim_input=40, dim_cell=256, dim_emb=64):
        super(DVectorModel, self).__init__()
        
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
        
    def forward(self, x, reshape=True):
        n_speaker = x.shape[0]
        if reshape:
            x = torch.reshape(x, (-1, x.shape[2], x.shape[3]))
            
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
        
        if reshape:
            x = torch.reshape(x, (n_speaker, -1, 512))
        return x
