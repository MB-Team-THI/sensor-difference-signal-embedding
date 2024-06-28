import torch.nn as nn

class model(nn.Module):
    def __init__(self,
                 idx=None,
                 name=None,
                 size=None,
                 n_params=None,
                 input=None,
                 output=None,
                 task=None,
                 description=None) -> None:
        super().__init__()
        self.idx = idx
        self.name = name
        self.size = size
        self.n_params = n_params
        self.input = input
        self.output = output
        self.task = task
        self.description = description

    def _get_description(self):
        description = {'idx':self.idx,
                       'name':self.name,
                       'size':self.size,
                       'n_params':self.n_params,
                       'input':self.input,
                       'output':self.output,
                       'task':self.task,
                       'description':self.description}
        return description
        
    def __str__(self):
        return self.name

