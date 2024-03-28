import torch
import torch.nn as nn
import torch_pruning as tp
from nets import generate_model

class Model(nn.Module):
    def __init__(self, model_name, model_path, in_channels=3, pruned_path=None) -> None:
        super().__init__()
        # self.model = NewAFFT(num_classes=14, pretrained=False, backbone='resnet18')
        self.model = generate_model(model_name, num_classes=14, in_channels=in_channels, pretrained=False, backbone='resnet18')
        
        if pruned_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            example_input = (torch.randn((1, 25, 416, 416)), 
                             torch.randn((1, 3, 416, 416)))
            DG = tp.DependencyGraph().build_dependency(self.model, example_input)
            state_dict = torch.load(pruned_path)
            DG.load_pruning_history(state_dict['pruning'])
        try:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        except:
            self.model = torch.load(model_path, map_location='cpu')
        
    
    def forward(self, *input):
        x = self.model(*input)
        x = x.permute(0, 2, 3, 1).softmax(dim=-1)
        return x