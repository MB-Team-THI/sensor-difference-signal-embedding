from torch import  optim
class optimizer(optim.Optimizer):
    def __init__(self,
                params,
                defaults,
                idx,
                name,
                description,
                ) -> None:
        self.id = idx
        self.name = name
        self.description = description
        super().__init__(params, defaults)

    def _get_description(self):
        description = {'id':self.id,
                        'name':self.name,
                        'description':self.description}
        return description