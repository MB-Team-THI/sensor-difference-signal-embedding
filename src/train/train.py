import abc


class Training(abc.ABC):
    def __init__(self) -> None:
        self.name = None
        self.size = None
        self.n_samples = None
        self.description = None

        # TODO
    @abc.abstractmethod
    def run_training(self, model, dataloader_train, loss_fc, optimizer, scheduler):
        """Implements the complete training task
        """
        pass

    def _get_description(self):
        #TODO
        description = {
            'name': self.name,
        }
        return description
