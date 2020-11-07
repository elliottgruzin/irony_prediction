import torch

class Dataset(torch.utils.data.Dataset):
  def __init__(self, identifier, label, path):
        self.label = label
        self.identifier = identifier
        self.path = path
  def __len__(self):
        'return number of samples'
        return len(self.identifier)

  def __getitem__(self, index):
        'return sample of data and corresponding label'
        # Select sample
        id = self.identifier[index]

        # Load data and get label
        X = torch.load('data/{}/'.format(self.path) + id + '.pt')
        y = self.label[id]

        return X, y
