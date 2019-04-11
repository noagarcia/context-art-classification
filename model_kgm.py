import torch.nn as nn
from torchvision import models


class KGM(nn.Module):
    # Inputs an image and ouputs the prediction for the class and the projected embedding into the graph space

    def __init__(self, num_class):
        super(KGM, self).__init__()

        # Load pre-trained visual model
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # Classifier
        self.classifier = nn.Sequential(nn.Linear(2048, num_class))

        # Graph space encoder
        self.nodeEmb = nn.Sequential(nn.Linear(2048, 128))


    def forward(self, img):

        visual_emb = self.resnet(img)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        pred_class = self.classifier(visual_emb)
        graph_proj = self.nodeEmb(visual_emb)

        return [pred_class, graph_proj]