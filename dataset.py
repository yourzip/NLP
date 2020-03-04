import random
import torch
import re
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer


def create_vocabulary(data):
  return list(set(data))


def generate_one_hot(voca, word):
  one_hot = [0] * len(voca)
  for i, w in enumerate(voca):
    if w == word:
      one_hot[i] = 1
  return one_hot


def stemming(data):
  print("stemming...")
  stemmer = SnowballStemmer('english')
  return [stemmer.stem(w) for w in data]


def remove_stopwords(data):
  print("removing stopwords...")
  stop_words = set(stopwords.words('english'))
  return [word for word in data if word not in stop_words]


def tokenize_text(data):
  print("tokenizing data...")
  return data.split()


def to_lower_case(data):
  print("to lower case...")
  return data.lower()


def letters_only(data):
  print("extracting letters only...")
  return re.sub('[^a-zA-Z]', ' ', data)


def remove_html_tags(data):
  print("removing html tags...")
  return BeautifulSoup(data, "html.parser").get_text()


def preprocessing(data, stopwords):
  data = remove_html_tags(data)
  data = letters_only(data)
  data = to_lower_case(data)
  data = tokenize_text(data)
  if stopwords:
    data = remove_stopwords(data)
  data = stemming(data)
  return (' '.join(data))


class WikiDataset(Dataset):
  def __init__(self, train, n=1, stopwords=False, path='./data', transform=None):
    path += '/wikidata'  # preprocessed data
    if not os.path.exists(path):
      print("no preprocessed data")
      f = open('./data/wikitext')  # raw data
      data = f.readline()
      f.close()
      data = preprocessing(data, stopwords)
      out = open('./data/wikidata', 'w')
      out.write(data)
      out.close()
    f = open(path)
    data = f.readline().split()
    f.close()
    data = data[:30]  # 너무 많아서..
    data_size = len(data) // 10 * 9
    empty = ["<empty>"] * (n-1)
    voca = create_vocabulary(data + empty)
    if train:
      data = data[:data_size]
    else:
      data = data[data_size:]
    self.empty = [generate_one_hot(voca, word) for word in empty]
    self.data = [generate_one_hot(voca, word) for word in data]
    self.n = n
    self.transform = transform

  def __len__(self):
    # len(self.empty) 더해야되나?
    return len(self.data)

  def __getitem__(self, idx):
    inputs = None
    labels = None
    idx = idx + (self.n-1)
    data = self.empty + self.data
    if self.n == 1:
      inputs = torch.FloatTensor(data[idx])
      # TODO.. 'labels' is none
    else:
      inputs = torch.FloatTensor(data[idx-(self.n-1):idx])
      labels = data[idx].index(1)
    if self.transform:
      inputs = self.transform(inputs)
    return inputs, labels


if __name__ == '__main__':
  wikiData = WikiDataset(train=True, n=6)
  dataloader = DataLoader(wikiData, batch_size=1,
                          shuffle=False, num_workers=1)
  for i, data in enumerate(dataloader):
    inputs, labels = data
    #print(inputs.size(), labels.size())
