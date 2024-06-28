import logging
import os
import pickle
from datetime import datetime
import numpy as np
import torch

from src.dataset.dataset import dataset

logging.basicConfig(filename="./logs.log",
                            filemode='a',
                            format='%(asctime)s, %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

class Experiment(object):
    def __init__(self,
                 meta_info = None,
                 dataset_param_train = None,
                 dataset_param_test = None,
                 trdataloader_param = None,
                 tedataloader_param = None,
                 model_param = None,
                 training_param = None,
                 evaluation_param = None,
                 optimizer_param = None,
                 scheduler_param = None,
                 loss_param = None,
                 run_name = None) -> None:
        
        super().__init__()
        self.meta_info = meta_info
        self.dataset_param_train = dataset_param_train
        self.dataset_param_test = dataset_param_test
        self.trdataloader_param = trdataloader_param
        self.tedataloader_param = tedataloader_param
        self.model_param = model_param
        self.training_param = training_param
        self.evaluation_param = evaluation_param
        self.optimizer_param = optimizer_param
        self.scheduler_param = scheduler_param
        self.loss_param = loss_param
        self.run_name = run_name

        self.trdataloader_param = {**self.trdataloader_param, "num_gpus":self.training_param["num_gpus"]}
        # Dataset and Dataloader
        self.dataset_train = []
        self.dataset_test = []
        for i in range(len(self.dataset_param_train)):
            self.dataset_train.append(dataset(**self.dataset_param_train[i]))
        #self.dataset_train = dataset(**self.dataset_param)
        # self.dataset_test = dataset(**{**self.dataset_param_test, "mode":"test"})
        for i in range(len(self.dataset_param_test)):
            self.dataset_test.append(dataset(**self.dataset_param_test[i]))
        logging.info(str(self.dataset_train[0]))
        logging.info(str(self.dataset_test[0]))

        dataloader_name = 'dataloader_' + str(trdataloader_param['idx'])
        import_str = "from src.dataloader.{0} import {0}".format(dataloader_name)
        exec(import_str) 
        self.dataloader_train = eval(dataloader_name + '(dataset=self.dataset_train,**self.trdataloader_param)')
        dataloader_name = 'dataloader_' + str(tedataloader_param['idx'])
        import_str = "from src.dataloader.{0} import {0}".format(dataloader_name)
        exec(import_str) 
        self.dataloader_test = eval(dataloader_name + '(dataset=self.dataset_test,**self.tedataloader_param)')
        # Model
        model_name = 'model_' + str(model_param['idx'])
        import_str = "from src.model.{0} import generate_model".format(model_name)
        exec(import_str) 
        self.model = eval('generate_model(**self.model_param)')        
        self.best_loss = 0
        # optimizer
        optimizer_name = 'optimizer_' + str(optimizer_param['idx'])
        import_str = "from src.train.optimizer.{0} import {0}".format(optimizer_name)
        exec(import_str) 
        self.optimizer = eval(optimizer_name+'(params = list(self.model.parameters()),**self.optimizer_param)')
        # Scheduler; Optional
        if scheduler_param is not None:
            scheduler_name = 'scheduler_' + str(scheduler_param['idx'])
            import_str = "from src.train.scheduler.{0} import {0}".format(
                scheduler_name)
            exec(import_str)
            dataloader_len = len(self.dataloader_train)
            epochs = self.dataloader_train.epochs
            self.scheduler_param = {
                **self.scheduler_param, 'dataloader_len': dataloader_len,
                'epochs': epochs
            }
            self.scheduler = eval(scheduler_name + '(**self.scheduler_param)')
        else:
            self.scheduler = None

        # loss
        loss_name = 'loss_' + str(loss_param['idx'])
        import_str = "from src.train.loss.{0} import {0}".format(loss_name)
        exec(import_str) 
        self.loss = eval(loss_name+'(**self.loss_param)')
        # eval
        eval_name = 'eval_' + str(evaluation_param['idx'])
        import_str = "from src.evaluation.{0} import {0}".format(eval_name)
        exec(import_str)
        self.eval_task = eval(eval_name+'(**self.evaluation_param)')
        self.epochs = self.trdataloader_param['epochs']
        # train
        train_name = 'train_' + str(training_param['idx'])
        import_str = "from src.train.{0} import {0}".format(train_name)
        exec(import_str)
        self.training_task = eval(train_name + '(**self.training_param)')

        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '8888'

    def __str__(self):
        task = "TASK: " + self.model.task if self.model.task != None else ""
        name = "NAME: " + self.model.name if self.model.name != None else ""
        desc = "DESCRIPTION: " + self.model.description if self.model.description != None else ""
        return name +" "  + task + " " +desc

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self):
        logging.info("Started training for experiment {}".format(str(self)))
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters()
                               if p.requires_grad)
        logging.info("Total model params: {:.0f}M (trainable: {:.0f}M)".format(
            total_params / 1e6, trainable_params / 1e6))

        self.training_task.run_training(self.model, self.dataloader_train, self.loss, self.optimizer, 
                                        self.scheduler, self.dataloader_test, self.eval_task, self.dataset_param_test, self.dataset_param_train, self.run_name)


    def evaluate(self):
        self.eval_task(self.model, self.dataset_param_test, self.dataloader_test, self.run_name, self.dataloader_train.epochs)



    def save_experiment_config(self,
                               path: str = None,
                               add_timestamp: bool = False):
        """Saves experiment config as pickle file to disk

        Args:
            path: Optional; path to which the config is saved. If no path is
                specified the filename will be the name specified in the
                meta_information
            add_timestamp: Optional; Whether or not to append a timestamp to the
                filename
        Note:
            Does not save model params
        """
        config = {
            "meta_info": self.meta_info,
            "dataset_param": self.dataset_param,
            "trdataloader_param": self.trdataloader_param,
            "tedataloader_param": self.tedataloader_param,
            "model_param": self.model_param,
            "training_param": self.training_param,
            "evaluation_param": self.evaluation_param,
            "optimizer_param": self.optimizer_param,
            "scheduler_param": self.scheduler_param,
            "loss_param": self.loss_param
        }
        if not path:
            os.makedirs("./saved_configs/", exist_ok=True)
            path = "./saved_configs/" + self.meta_info["name"]
        if add_timestamp:
            timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            path += "_{}".format(timestamp)
        if not path.endswith(".pkl"):
            path += ".pkl"
        with open(path, 'wb') as f:
            pickle.dump(config, f)
        logging.info("Saved config {} \n to file {}".format(config, path))

    @classmethod
    def from_config_file(cls, path: str):
        """Factory for creating an Experiment with a previously saved config

        Args:
            path: path to config

        Returns:
            Experiment
        """
        with open(path, 'rb') as f:
            return cls(**pickle.load(f))
