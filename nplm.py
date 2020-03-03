import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.corpus import stopwords
from torch.utils.data import Dataset, DataLoader
from dataset import WikiDataset
import random
import os


m = 100  # embedding size of word
n = 6  # n-gram's n
h = 60  # hidden-layer output size
batch_size = 1
MODEL_PATH = './models'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NPLM(nn.Module):
  # input_size: (n-1, |V|)
  def __init__(self, size):
    super(NPLM, self).__init__()
    self.fc1 = nn.Linear(size, m, bias=False)
    self.fc2 = nn.Linear(m*(n-1), h)
    self.fc3 = nn.Linear(h, size)
    self.fc4 = nn.Linear(m*(n-1), size, bias=False)

  def forward(self, x):
    x = self.fc1(x)
    x = x.view(-1, m*(n-1))
    y = torch.tanh(self.fc2(x))
    x = self.fc3(y) + self.fc4(x)
    x = F.softmax(x, dim=1)
    return x

  def predict(self, dataloader, batch_size=1):
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    acc = 0
    data_size = 0
    for i, data in enumerate(dataloader):
      inputs, labels = data
      outputs = self.forward(inputs)
      for j in range(batch_size):
        values, indices = torch.max(outputs[j], 0)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        if indices.item() == labels[j].item():
          acc += 1
      data_size += 1
    test_acc = acc/data_size
    test_loss = test_loss/data_size
    return test_loss, test_acc


def run():
  if not os.path.exists('models'):
    os.makedirs('models')
  print("loading data...")
  wiki_data = WikiDataset(num=500, n=4)
  train_data = wiki_data[:400]
  val_data = wiki_data[400:450]
  test_data = wiki_data[450:]
  train_dataloader = DataLoader(train_data, batch_size=batch_size,
                                shuffle=False, num_workers=1)
  val_dataloader = DataLoader(val_data, batch_size=batch_size,
                              shuffle=False, num_workers=1)
  test_dataloader = DataLoader(test_data, batch_size=batch_size,
                               shuffle=False, num_workers=1)
  print("done")
  nplm = NPLM(len(voca)).to(device)
  # criterion = nn.MSELoss()
  # loss함수랑 optimizer를 모델 안에 넣으면 안되나?
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(nplm.parameters(), lr=0.001, momentum=0.9)

  loss_p = float("Inf")
  for epoch in range(10000):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
      inputs, labels = data
      outputs = nplm(inputs)
      optimizer.zero_grad()
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      val_loss, val_acc = nplm.predict(data_val, batch_size=batch_size)
      print('[epoch: %d] train_loss: %f val_loss: %f val_acc: %f' %
            (epoch + 1, running_loss / len(wikiData), val_loss, val_acc))
      if running_loss < loss_p:
        model_name = MODEL_PATH + '/nplm_epoch' + \
            str(epoch+1) + '_acc' + str(val_acc) + '.pth'
        torch.save(nplm.state_dict(), model_name)
        print("    model saved.")
        test_loss, test_acc = nplm.predict(data_test, batch_size=batch_size)
        print('    test_loss: %f test_acc: %f' % (test_loss, test_acc))
        loss_p = running_loss


if __name__ == '__main__':
  run()
