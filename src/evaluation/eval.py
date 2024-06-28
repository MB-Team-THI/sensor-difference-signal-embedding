class eval(object):
    def __init__(self,
                 idx=None,
                 name=None,
                 input_=None,
                 output=None,
                 description=None) -> None:
        super().__init__()
        self.idx = idx
        self.name = name
        self.input = input_
        self.output = output
        self.description = description

    def _get_description(self):
        description = {'idx':self.idx,
                       'name':self.name,
                       'input':self.input,
                       'output':self.output,
                       'description':self.description}
        return description

