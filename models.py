from torch import nn

class AlexNet(nn.Module):
    def __init__(self):
        #super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
        	nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 3),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        predict = self.classifier(x)
        return predict


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.classifier = nn.Sequential(
	        nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1),
	        nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
	        nn.ReLU(inplace=True),
	        nn.MaxPool2d(2, 2),
	        nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
	        nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
	        nn.ReLU(inplace=True),
	        nn.MaxPool2d(2, 2),
	        nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
	        nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
	        nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
	        nn.ReLU(inplace=True),
	        nn.MaxPool2d(2, 2),
	        nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
	        nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
	        nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
	        nn.ReLU(inplace=True),
	        nn.MaxPool2d(2, 2),
	        nn.Flatten(),
	        nn.Linear(512*8*8, 4096),
	        nn.ReLU(inplace=True),
	        nn.Dropout(0.2),
	        nn.Linear(4096, 10),
	        nn.Linear(10, 3)
	    )

    def forward(self, x):
    	predict = self.classifier(x)
    	return predict


class ReducedVGG(nn.Module):
    def __init__(self):
        super(ReducedVGG, self).__init__()
        self.classifier = nn.Sequential(
	        nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1),
	        nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
	        nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
	        nn.ReLU(inplace=True),
	        nn.MaxPool2d(2, 2),   # size, stride
	        nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
	        nn.ReLU(inplace=True),
	        nn.MaxPool2d(2, 2),
	        nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
	        nn.ReLU(inplace=True),
	        nn.MaxPool2d(2, 2),
	        nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
	        nn.ReLU(inplace=True),
	        nn.MaxPool2d(2, 2),
	        nn.Flatten(),
	        nn.Linear(512*8*8, 4096),
	        nn.ReLU(inplace=True),
	        nn.Dropout(0.2),
	        nn.Linear(4096, 10),
	        nn.Linear(10, 3)
	    )

    def forward(self, x):
    	predict = self.classifier(x)
    	return predict


class FourConv(nn.Module):
    def __init__(self):
        super(FourConv, self).__init__()
        self.classifier = nn.Sequential(
	        nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1),
	        nn.ReLU(inplace=True),
	        nn.MaxPool2d(2, 2),
	        nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
	        nn.ReLU(inplace=True),
	        nn.MaxPool2d(2, 2),
	        nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
	        nn.ReLU(inplace=True),
	        nn.MaxPool2d(2, 2),
	        nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
	        nn.ReLU(inplace=True),
	        nn.MaxPool2d(2, 2),
	        nn.Flatten(),
	        nn.Linear(512*8*8, 4096),
	        nn.ReLU(inplace=True),
	        nn.Dropout(0.2),
	        nn.Linear(4096, 10),
	        nn.Linear(10, 3)
	    )

    def forward(self, x):
    	predict = self.classifier(x)
    	return predict