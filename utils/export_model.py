import torch
import torch.nn as nn
import torch_pruning as tp
from nets.newafft import NewAFFT

class Model(nn.Module):
    def __init__(self, model_path, pruned_path=None) -> None:
        super().__init__()
        self.model = NewAFFT(num_classes=14, pretrained=False, backbone='resnet18')
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
        
    
    def forward(self, spec, rgb):
        x = self.model(spec, rgb)
        x = x.permute(0, 2, 3, 1).softmax(dim=-1)
        return x