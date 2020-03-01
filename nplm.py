import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.corpus import stopwords
import random
import os


m = 100  # embedding size of word
n = 6  # n-gram's n
h = 60  # hidden-layer output size
batch_size = 1
MODEL_PATH = './models'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def tokenize_text(txt):
  return txt.split()


def remove_stopwords(data):
  print("removing stopwords...")
  stop_words = set(stopwords.words('english'))
  return [word for word in data if word not in stop_words]


def load_wikitext(path="./data/wikitext"):
  f = open(path)
  data = f.readline()
  return remove_stopwords(tokenize_text(data))


def create_vocabulary(data):
  return list(set(data))


def generate_one_hot(voca, word):
  one_hot = [0] * len(voca)
  a = False
  for i, w in enumerate(voca):
    if w == word:
      one_hot[i] = 1
      a = True
  if not a:
    print(word)
  return one_hot


def dataloader(data, batch_size=1):
  batch_range = list(range((n-1), len(data)-(n-1)))
  batch_idxs = random.sample(batch_range, batch_size)
  inputs = torch.FloatTensor([data[i-(n-1):i] for i in batch_idxs]).to(device)
  # labels = torch.FloatTensor([data[i] for i in batch_idxs]).to(device)
  labels = torch.LongTensor([data[i].index(1) for i in batch_idxs]).to(device)
  return inputs, labels


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

  def predict(self, data, batch_size=1):
    data_size = len(data)-2*(n-1)
    till = 0
    if len(data) % batch_size != 0:
      till += int(data_size/batch_size)+1
    else:
      till += data_size//batch_size
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    acc = 0
    for i in range(till):
      inputs, labels = dataloader(data, batch_size)
      outputs = self.forward(inputs)
      for i in range(batch_size):
        values, indices = torch.max(outputs[i], 0)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        if indices.item() == labels[i].item():
          acc += 1
    test_acc = acc/data_size
    test_loss = test_loss/data_size
    if test_acc >= 1.0:
      test_acc = 1.0
    return test_loss, test_acc


if not os.path.exists('models'):
  os.makedirs('models')
print("loading data...")
data = load_wikitext()  # 81534361 words
data = data[:110000]
print("creating vocabulary...")
voca = create_vocabulary(data)
print("generating one-hot...")
data = [generate_one_hot(voca, word) for word in data]
data_test = data[81000000:]
data_val = data[80000000:81000000]
data = data[:80000000]
print("done")
nplm = NPLM(len(voca)).to(device)
# criterion = nn.MSELoss()
# loss함수랑 optimizer를 모델 안에 넣으면 안되나?
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(nplm.parameters(), lr=0.001, momentum=0.9)

loss_p = float("Inf")
for epoch in range(10000):
  running_loss = 0.0
  till = int(len(data)/batch_size)+1
  for i in range(till):
    inputs, labels = dataloader(data, batch_size)
    outputs = nplm(inputs)
    optimizer.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
  val_loss, val_acc = nplm.predict(data_val, batch_size=batch_size)
  print('[epoch: %d] train_loss: %f val_loss: %f val_acc: %f' %
        (epoch + 1, running_loss / (batch_size * till), val_loss, val_acc))
  if running_loss < loss_p:
    model_name = MODEL_PATH + '/nplm_epoch' + \
        str(epoch+1) + '_acc' + str(val_acc) + '.pth'
    torch.save(nplm.state_dict(), model_name)
    print("    model saved.")
    test_loss, test_acc = nplm.predict(data_test, batch_size=batch_size)
    print('    test_loss: %f test_acc: %f' % (test_loss, test_acc))
    loss_p = running_loss
