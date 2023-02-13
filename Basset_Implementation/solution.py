# This script contains the helper functions you will be using for this assignment

import os
import random

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader, dataloader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BassetDataset(Dataset):
    """
    BassetDataset class taken with permission from Dr. Ahmad Pesaranghader

    We have already processed the data in HDF5 format: er.h5
    See https://www.h5py.org/ for details of the Python package used

    We used the same data processing pipeline as the paper.
    You can find the code here: https://github.com/davek44/Basset
    """

    # Initializes the BassetDataset
    def __init__(self, path='./data/', f5name='er.h5', split='train', transform=None):
        """
        Args:
            :param path: path to HDF5 file
            :param f5name: HDF5 file name
            :param split: split that we are interested to work with
            :param transform (callable, optional): Optional transform to be applied on a sample
        """

        self.split = split

        split_dict = {'train': ['train_in', 'train_out'],
                      'test': ['test_in', 'test_out'],
                      'valid': ['valid_in', 'valid_out']}

        assert self.split in split_dict, "'split' argument can be only defined as 'train', 'valid' or 'test'"

        # Open hdf5 file where one-hoted data are stored
        self.dataset = h5py.File(os.path.join(path, f5name.format(self.split)), 'r')

        # Keeping track of the names of the target labels
        self.target_labels = self.dataset['target_labels']

        # Get the list of volumes
        self.inputs = self.dataset[split_dict[split][0]]
        self.outputs = self.dataset[split_dict[split][1]]

        self.ids = list(range(len(self.inputs)))
        if self.split == 'test':
            self.id_vars = np.char.decode(self.dataset['test_headers'])

    def __getitem__(self, i):
        """
        Returns the sequence and the target at index i

        Notes:
        * The data is stored as float16, however, your model will expect float32.
          Do the type conversion here!
        * Pay attention to the output shape of the data.
          Change it to match what the model is expecting
          hint: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        * The target must also be converted to float32
        """
        idx = self.ids[i]

        # Sequence & Target
        output = {'sequence': None, 'target': None}

        
        sequence = torch.Tensor(self.inputs[idx]).permute(1, 2, 0)
        target = torch.Tensor(self.outputs[idx])

        output['sequence'] = sequence
        output['target'] = target

        return output

    def __len__(self):

        # WRITE CODE HERE
        return len(self.inputs)

    def get_seq_len(self):
        """
        Answer to Q1 part 2
        """
        # WRITE CODE HERE
        return self.inputs[0].shape[-1]

    def is_equivalent(self):
        """
        Answer to Q1 part 3
        """
        # WRITE CODE HERE
        sequence_length = self.get_seq_len()
        return self.inputs[0].shape == (4,1,sequence_length)


class Basset(nn.Module):
    """
    Basset model
    Architecture specifications can be found in the supplementary material
    You will also need to use some Convolution Arithmetic
    """

    def __init__(self):
        super(Basset, self).__init__()

        self.dropout = 0.3
        self.num_cell_types = 164
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout_ = nn.Dropout(self.dropout)
        self.conv1 = nn.Conv2d(1, 300, (19, 4), stride=(1, 1), padding=(9, 0))
        self.conv2 = nn.Conv2d(300, 200, (11, 1), stride=(1, 1), padding=(5, 0))
        self.conv3 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(4, 0))
        self.bn1 = nn.BatchNorm2d(300)
        self.bn2 = nn.BatchNorm2d(200)
        self.bn3 = nn.BatchNorm2d(200)
        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.maxpool2 = nn.MaxPool2d((4, 1))
        self.maxpool3 = nn.MaxPool2d((4, 1))
        self.fc1 = nn.Linear(13*200, 1000)
        self.bn4 = nn.BatchNorm1d(1000)###
        self.fc2 = nn.Linear(1000, 1000)###
        self.bn5 = nn.BatchNorm1d(1000)###
        self.fc3 = nn.Linear(1000, self.num_cell_types)###

    def forward(self, x):

        # WRITE CODE HERE
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout_(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.dropout_(x)
        x = self.fc3(x)


        return x


def compute_fpr_tpr(y_true, y_pred):
    """
    Computes the False Positive Rate and True Positive Rate
    Args:
        :param y_true: groundtruth labels (np.array of ints)
        :param y_pred: model decisions (np.array of ints)

    :Return: dict with tpr, fpr (values are floats)
    """
    output = {'fpr': 0., 'tpr': 0.}

    # WRITE CODE HERE
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tp = np.sum((y_pred == 1) & (y_true == 1))

    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)

    output = {'fpr': fpr, 'tpr': tpr}
    return output


def compute_fpr_tpr_dumb_model():
    """
    Simulates a dumb model and computes the False Positive Rate and True Positive Rate

    :Return: dict with tpr_list, fpr_list.
             These lists contain the tpr and fpr for different thresholds
             fpr and tpr values in the lists should be floats
             Order the lists such that:
                 output['fpr_list'][0] corresponds to k=0.
                 output['fpr_list'][1] corresponds to k=0.05 
                 ...
            Do the same for output['tpr_list']
             
    """
    y_true = np.random.randint(2, size = 1000)
    y_pred = np.random.uniform(0,1,1000)
    k = np.arange(0,1,0.05)
    
    fpr_list = []
    tpr_list = []
    
    for j in k:
      y_pred_treshold = np.zeros(len(y_pred))
      for i in range(len(y_pred)):
        if y_pred[i] > j:
          y_pred_treshold[i]= 1
      
      result = compute_fpr_tpr(y_true, y_pred_treshold)
      fpr_list.append(result['fpr'])
      tpr_list.append(result['tpr'])

    output = {'fpr_list': fpr_list, 'tpr_list': tpr_list}


    return output


def compute_fpr_tpr_smart_model():
    """
    Simulates a smart model and computes the False Positive Rate and True Positive Rate

    :Return: dict with tpr_list, fpr_list.
             These lists contain the tpr and fpr for different thresholds
             fpr and tpr values in the lists should be floats
             Order the lists such that:
                 output['fpr_list'][0] corresponds to k=0.
                 output['fpr_list'][1] corresponds to k=0.05 
                 ...
            Do the same for output['tpr_list']
    """
    output = {'fpr_list': [], 'tpr_list': []}

    y_true = np.random.randint(2, size = 1000)
    k = np.arange(0,1.0,0.05)

    y_pred =([np.random.uniform(0.4,1,1)[0] if i>0.5 else np.random.uniform(0,0.6,1)[0] for i in y_true])
    

    fpr_list = []
    tpr_list = []
    
    for j in k:
      y_pred_treshold = np.zeros(len(y_pred))
      for i in range(len(y_pred)):
        if y_pred[i] > j:
          y_pred_treshold[i]= 1
      
      result = compute_fpr_tpr(y_true, y_pred_treshold)
      fpr_list.append(result['fpr'])
      tpr_list.append(result['tpr'])
  
    output = {'fpr_list': fpr_list, 'tpr_list': tpr_list}


    return output


def compute_auc_both_models():
    """
    Simulates a dumb model and a smart model and computes the AUC of both

    :Return: dict with auc_dumb_model, auc_smart_model.
             These contain the AUC for both models
             auc values in the lists should be floats
    """
    output = {'auc_dumb_model': 0., 'auc_smart_model': 0.}

    dumb_output = compute_fpr_tpr_dumb_model()
    fpr_list_dumb = dumb_output['fpr_list']
    tpr_list_dumb = dumb_output['tpr_list']

    smart_output = compute_fpr_tpr_smart_model()
    fpr_list_smart = smart_output['fpr_list']
    tpr_list_smart = smart_output['tpr_list']

    auc_dumb = 0
    auc_smart = 0
    for k in range(len(fpr_list_dumb)-1):
      auc_dumb += + (fpr_list_dumb[k]- fpr_list_dumb[k + 1]) * tpr_list_dumb[k]
      auc_smart += + (fpr_list_smart[k]- fpr_list_smart[k + 1]) * tpr_list_smart[k]


    output = {'auc_dumb_model': auc_dumb, 'auc_smart_model': auc_smart}


    return output


def compute_auc_untrained_model(model, dataloader, device):
    """
    Computes the AUC of your input model

    Dont forget to re-apply your output activation!

    :Return: dict with auc_dumb_model, auc_smart_model.
             These contain the AUC for both models
             auc values should be floats

    Make sure this function works with arbitrarily small dataset sizes!
    
    """


    output = {'auc': 0.}
    sigmoid = torch.nn.Sigmoid()
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    k = np.arange(0,1,0.05)
    model = model.to(device)
    y_model = []
    y = []
    for data in dataloader:
      

      inputs, labels = data['sequence'], data['target']
      y = np.append(y, labels.view(-1).detach())
      inputs = inputs.to(device)

      y_hat_batch = sigmoid(model(inputs).view(-1))


      y_model = np.append(y_model, y_hat_batch.cpu().detach())
    
    output = compute_auc(y, y_model)

    return output


def compute_auc(y_true, y_model):
    """
    Computes area under the ROC curve
    auc returned should be float
    Args:
        :param y_true: groundtruth labels (np.array of ints)
        :param y_pred: model decisions (np.array of ints)
    """
    output = {'auc': 0.}
    
    # WRITE CODE HERE
    partition = np.arange(0,1.0,0.05)
    
    fpr_list = []
    tpr_list = []
    
    for j in partition:
      y_pred_treshold = np.zeros(len(y_model))
      for i in range(len(y_model)):
        if y_model[i] > j:
          y_pred_treshold[i]= 1

      result = compute_fpr_tpr(y_true, y_pred_treshold)
      fpr_list.append(result['fpr'])
      tpr_list.append(result['tpr'])
    
    
    # import matplotlib.pyplot as plt

    # #plot roc:
    # plt.plot(fpr_list, tpr_list)
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()  


    auc_left = 0
    auc_right = 0
    for k in range(len(partition)-1):
      auc_left += np.abs((fpr_list[k]- fpr_list[k + 1]) * tpr_list[k])
      auc_right += np.abs(tpr_list[k+1]*(fpr_list[k+1]-fpr_list[k]))
    auc = (auc_right + auc_left) / 2.
    output['auc'] = auc
    return output


def get_critereon():
    """
    Picks the appropriate loss function for our task
    criterion should be subclass of torch.nn
    """

    # WRITE CODE HERE
    critereon = torch.nn.BCEWithLogitsLoss()
    

    return critereon


def train_loop(model, train_dataloader, device, optimizer, criterion):
    """
    One Iteration across the training set
    Args:
        :param model: solution.Basset()
        :param train_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)

    :Returns: output dict with keys: total_score, total_loss
    values of each should be floats
    (if you want to record losses or scores within the, you may print them to screen)

    Make sure your loop works with arbitrarily small dataset sizes!
    """

    output = {'total_score': 0.,
              'total_loss': 0.}
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # WRITE CODE HERE
    total_loss = 0
    correct = 0
    total = 0
    output_list = []
    label_list = []
    counter = 0
    for batch in train_dataloader:
      
      optimizer.zero_grad()

      inputs, label = batch['sequence'], batch['target']
      inputs = inputs.to(device)
      
      model = model.to(device)
      outputs = model(inputs)
      loss = criterion(outputs.cpu(), label)
      loss.backward()
      optimizer.step()


      total = label.size(0)
      total_loss += loss.sum().data.cpu() * total
      counter += total

      outputs = outputs.view(-1).cpu().detach()
      label = label.view(-1).cpu().detach()
      output_list = np.append(output_list, outputs)
      label_list = np.append(label_list, label)
      
    
    total_score = compute_auc(label_list, output_list)['auc']


    
    output['total_loss'] = total_loss / counter
    output['total_score'] = total_score

#cite: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


    return output['total_score'], output['total_loss']


def valid_loop(model, valid_dataloader, device, optimizer, criterion):
    """
    One Iteration across the validation set
    Args:
        :param model: solution.Basset()
        :param valid_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)

    :Returns: output dict with keys: total_score, total_loss
    values of each should be floats
    (if you want to record losses or scores within the, you may print them to screen)

    Make sure your loop works with arbitrarily small dataset sizes!
    """

    output = {'total_score': 0.,
              'total_loss': 0.}

    # WRITE CODE
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # WRITE CODE HERE
    total_loss = 0
    correct = 0
    total = 0
    output_list = []
    label_list = []
    counter = 0
    model.eval()
    for batch in valid_dataloader:
      
      inputs, label = batch['sequence'], batch['target']
      inputs = inputs.to(device)
      
      model = model.to(device)
      outputs = model(inputs)
      loss = criterion(outputs.cpu(), label)


      total = label.size(0)
      total_loss += loss.sum().data.cpu() * total
      counter += total

      outputs = outputs.view(-1).cpu().detach()
      label = label.view(-1).cpu().detach()
      output_list = np.append(output_list, outputs)
      label_list = np.append(label_list, label)
    
    total_score = compute_auc(label_list, output_list)['auc']


    
    output['total_loss'] = total_loss / counter
    output['total_score'] = total_score
    
    


    return output['total_score'], output['total_loss']
