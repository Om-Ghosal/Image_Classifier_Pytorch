from torch import nn,save,load
import torch
from PIL import Image
from torchvision.transforms import ToTensor

class ImageClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(64*(28-6)*(28-6), 10)  
        )

    def forward(self, x): 
        return self.model(x)
    
clf=ImageClassifier()
if __name__ == '__main__':
    with open('model_state.pt','rb') as f:
        clf.load_state_dict(load(f))

    img= Image.open('img_1.jpg')
    img_tensor=ToTensor()(img).unsqueeze(0)

    print(torch.argmax(clf(img_tensor)))
