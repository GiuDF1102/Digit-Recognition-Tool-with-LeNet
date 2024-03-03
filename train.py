from torch import nn
from data.mnist import MNIST
from torch.utils.data import DataLoader
from torch import cuda, optim, nn, device, save
from model.LeNet5 import LeNet5


def train(model, optimizer, dataloader_train, num_epochs, device):
  cost = nn.CrossEntropyLoss()
  total_step = len(dataloader_train)

  for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(dataloader_train):
      images = images.to(device)
      labels = labels.to(device)

      outputs = model(images)
      loss = cost(outputs, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if (i+1) % 400 == 0:
          print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


         
if __name__ == '__main__':
   
  device = device('cuda' if cuda.is_available() else 'cpu')
  num_epochs = 10

  train_dataset = MNIST(mode='train')

  dataloader_train = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)

  model = LeNet5()
  if cuda.is_available():
      model = nn.DataParallel(model).cuda()

  optimizer = optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=1e-4)

  #train
  train(model, optimizer, dataloader_train, num_epochs, device)

  #save model
  save(model.state_dict(), 'lenet5.pth')