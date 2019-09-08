import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torch.nn.functional as F

def train_val_split(train, train_file, val_file):

    train_data = pd.read_csv(train)
    train_set, val_set = train_test_split(train_data, test_size=0.3)
    # save train set
    train_set.to_csv(train_file, index=False)
    # save val set
    val_set.to_csv(val_file, index=False)

    print('train_data.shape', train_data.shape)
    print('train_set.shape', train_set.shape)
    print('val_set.shape', val_set.shape)


def data_tf(x):
    # image preprocess to Gaussian distribution from -1 to 1
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5
    x = torch.from_numpy(x)
    return x


class MyMINST(torch.utils.data.Dataset):
    def __init__(self, data_text, train=True, transform=data_tf, target_transform=None):
        self.data = pd.read_csv(data_text)
        self.transform = transform
        self.train = train
        if self.train:
            self.X = self.data.iloc[:, 1:]
            self.X = np.array(self.X)
            self.X = self.X.reshape(self.X.shape[0], 1, 28, 28)
            self.Y = self.data.iloc[:, 0]
            self.Y = np.array(self.Y)
        else:
            self.X = self.data
            self.X = np.array(self.X)
            self.X = self.X.reshape(self.X.shape[0], 1, 28, 28)

    def __getitem__(self, item):
        im = torch.tensor(self.X[item], dtype=torch.float)
        if self.transform is not None:
            im = self.transform(im)
        if self.train:
            label = torch.tensor(self.Y[item], dtype=torch.long)
            return im, label
        else:
            return im

    def __len__(self):
        return len(self.data)


class CNet(torch.nn.Module):
    def __init__(self):
        super(CNet, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        # origin pic is 28*28
        # after first conv2d  (28+4-5)+1 = 28
        # after first maxpool  14
        # after second conv2d (14+4-5)+1 = 14
        # after first maxpool  7
        self.fc = torch.nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.conv(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def train_model(model, num_epoch, data_loader, criterion, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                num_epoch, (batch_idx + 1) * len(data), len(data_loader.dataset),
                100. * (batch_idx + 1)/ len(data_loader), loss.item()
            ))


def eval_model(model, data_loader):
    model.eval()
    loss = 0
    correct = 0

    for data, target in data_loader:
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        output = model(data)

        loss += F.cross_entropy(output, target, size_average=False).item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    loss /= len(data_loader.dataset)

    print('\nAverage Val Loss: {:.4f}, Val Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(data_loader.dataset),
        (100.*correct)/len(data_loader.dataset)
    ))


def make_prediction(model, data_loader):
    model.eval()
    test_pred = torch.LongTensor()

    for i, data in enumerate(data_loader):
        if torch.cuda.is_available():
            data = data.cuda()
        output = model(data)
        preds = output.cpu().data.max(1, keepdim=True)[1]
        test_pred = torch.cat((test_pred, preds), dim=0)

    return test_pred

if __name__ == '__main__':
    # split train and test set, run only once
    # train_val_split('data/train.csv', 'data/train_set.csv', 'data/val_set.csv')
    X_train = MyMINST('data/train_set.csv', train=True, transform=data_tf)
    X_val = MyMINST('data/val_set.csv', train=True, transform=data_tf)
    X_test = MyMINST('data/test.csv', train=False, transform=data_tf)

    train_data = DataLoader(X_train, batch_size=64, shuffle=True)
    val_data = DataLoader(X_val, batch_size=64, shuffle=False)
    test_data = DataLoader(X_test, batch_size=1000, shuffle=False)

    learning_rate = 0.001
    num_epoch = 25
    net = CNet()
    net.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(num_epoch):
        train_model(net, epoch, train_data, criterion, optimizer)
        eval_model(net, val_data)

    test_set_preds = make_prediction(net, test_data)

    submission_df = pd.read_csv('data/sample_submission.csv')

    submission_df['Label'] = test_set_preds.numpy().squeeze()
    submission_df.to_csv('data/submission.csv', index=False)



