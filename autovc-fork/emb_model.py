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
        
class FullModel(torch.nn.Module):
    def __init__(self, nclasses=16):
        super(FullModel, self).__init__()
        
        self.embed = XVectorConv2D()
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
