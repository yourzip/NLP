import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from nltk.corpus import stopwords


def tokenize_text(txt):
  return txt.split()


def remove_stopwords(data):
  stop_words = set(stopwords.words('english'))
  return [word for word in data if word not in stop_words]


def create_vocabulary(data):
  return list(set(data))


def generate_one_hot(voca, word):
  one_hot = [0] * len(voca)
  for i, w in enumerate(voca):
    if w == word:
      one_hot[i] = 1
  return one_hot


class WikiDataset(Dataset):
  def __init__(self, n=1, path='./data/wikitext', num=0, transform=None):
    f = open(path)
    data = f.readline()
    f.close()
    data_raw = remove_stopwords(tokenize_text(data))  # 81534361 words
    if num > 0:
      data_raw = data_raw[:num]
    empty = ["<empty>"] * (n-1)
    voca = create_vocabulary(data_raw + empty)
    self.empty = [generate_one_hot(voca, word) for word in empty]
    self.data = [generate_one_hot(voca, word) for word in data_raw]
    self.n = n
    self.transform = transform

  def __len__(self):
    # len(self.empty) 더해야되나?
    return len(self.data)

  def __getitem__(self, idx):

    idx = idx + (self.n-1)
    data = self.empty + self.data
    inputs = None
    labels = None
    if self.n == 1:
      inputs = torch.FloatTensor(data[idx])
      # TODO.. 'labels' is none
    else:
      inputs = torch.FloatTensor(data[idx-(self.n-1):idx])
      labels = torch.LongTensor([data[idx].index(1)])
    if self.transform:
      inputs = self.transform(inputs)
    return inputs, labels


if __name__ == '__main__':
  wikiData = WikiDataset(num=500, n=4)
  dataloader = DataLoader(wikiData, batch_size=3,
                          shuffle=False, num_workers=1)
  for i, data in enumerate(dataloader):
    inputs, labels = data
    print(inputs.size(), labels.size)
