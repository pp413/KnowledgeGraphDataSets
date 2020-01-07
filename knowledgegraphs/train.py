from tqdm import tqdm
from abc import ABC, abstractmethod

class Train_Base(ABC):
    
    def __init__(self, model, optimizer, batch_size, criterion, device, verbose, **kwargs):
        
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.criterion = criterion
        self.device = device
        self.verbose = verbose
        
        self.model.to(device)
        
        self.__dict__.update(kwargs)
    
    @ abstractmethod
    def epoch(self, samples):
        pass


class TrainingWithCriterion(Train_Base, ABC):
        
    def epoch(self, samples, criterion):
        samples = torch.from_numpy(samples)
        actual_samples = examples[torch.randperm(samples.shape[0]), :]
        with tqdm.tqdm(total=samples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < samples.shape[0]:
                input_batch = actual_samples[
                    b_begin:b_begin + self.batch_size
                ].to(self.device)

                loss = self.loss(input_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{loss.item():.2f}')
    
    @ abstractmethod
    def loss(batch_data):
        pass


class Training_CrossEntropyLoss(TrainingWithCriterion):
    
    def __init__(self, model, optimizer, batch_size, device, verbose):
        super(Training_CrossEntropyLoss, self).__init__(
            model, optimizer, batch_size, nn.CrossEntropyLoss(reduction='mean'), device, verbose,
        )
    
    def loss(batch_data):
        predictions = self.model(batch_data)
        truth = batch_data[:, -1]
        return self.criterion(predictions, truth)
