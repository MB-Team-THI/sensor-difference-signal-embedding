import torch
import torch.nn as nn

class loss(nn.Module):
    def __init__(self,idx,name,description,input_,output) -> None:
        self.idx = idx,
        self.name = name
        self.description = description
        self.input = input_
        self.output = output
        super().__init__()
    def _get_description(self):
        description = { 'idx':self.idx,
                        'name':self.name,
                        'description':self.description,
                        'input':self.input,
                        'output':self.output}
        return description