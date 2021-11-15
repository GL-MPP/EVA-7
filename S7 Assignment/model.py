class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.depthwise_separable_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, groups=3, padding=1, bias=False,dilation=1),
            nn.BatchNorm2d(3),
            #nn.Dropout(.05),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, bias=False),
            #nn.Dropout(.05),
            nn.BatchNorm2d(32),
            nn.Dropout(.05),
            nn.ReLU(),
        )
        
        
        self.depthwise_separable_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, groups=32, padding=1, dilation=1,
                      bias=False),
            nn.BatchNorm2d(32),
            #nn.Dropout(.10),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            #nn.Dropout(.10),
            nn.ReLU(),
        )
        
        
        self.transposeconvblock = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False,
                               dilation=4),
            nn.BatchNorm2d(128),
            #nn.Dropout(.10),
            nn.ReLU(),

        )
        
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(3, 3), padding=0, bias=False, dilation=4),
            nn.BatchNorm2d(10),
            #nn.Dropout(.5),
            nn.ReLU()
        )
        
               
        
        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=18))
        
        

    def forward(self, x):
        
        x = self.depthwise_separable_conv1(x)
        x = self.depthwise_separable_conv2(x)
        x = self.transposeconvblock(x)
        x = self.convblock(x)
        
               
        x = self.gap(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=-1)