#Pytorch-lightning setup
import sys
sys.path.append('../')
from lib.medzoo.Unet3D import UNet3D
from lib.losses3D.basic import compute_per_channel_dice, expand_as_one_hot
import torch
import pytorch_lightning as pl

class TumourSegmentation(pl.LightningModule):
  def __init__(self, learning_rate):
    super().__init__()
    self.model =  UNet3D(in_channels=4, n_classes=2, base_n_filter=8) #.cuda()
    self.learning_rate = learning_rate
  
  def forward(self,x):
  #  x=x.half()

    f = self.model.forward(x)

  #  print('Done forward step!')
    return f

  def training_step(self, batch, batch_idx):
    x, y = batch
    x = torch.unsqueeze(x, axis=0)
    y = torch.unsqueeze(y, axis=0)
    #print(x.shape)

    y_hat = self.forward(x)

    #plt.imshow(y_hat[0, 0, 120], cmap='')
    #plt.imshow(y_hat.cpu()[0, 1, 120])
    #plt.imshow(y_hat[0, 2, 120])
    #plt.imshow(y_hat[0, 3, 120])
    #plt.show()

    shape = list(y.size())
    shape[1] = 2
    zeros = torch.zeros(shape).cuda()

    zeros[:, 0][torch.torch.squeeze(y == 1, dim=1)] = 1
    zeros[:, 0][torch.torch.squeeze(y == 4, dim=1)] = 1
    zeros[:, 1][torch.torch.squeeze(y == 2, dim=1)] = 1
    #for i, label_n in enumerate([1,2,4]):
 #     zeros[:, i][torch.squeeze(y == label_n, dim=1)] = 1

  # basic mean of all channels for now
    loss = -1*compute_per_channel_dice(y_hat, zeros)
    loss[loss != loss] = 0
    print('Training loss: ')
    print(loss)
    loss = torch.sum(loss)
    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    x = torch.unsqueeze(x, axis=0)
    y = torch.unsqueeze(y, axis=0)
    #print(x.shape)

    y_hat = self.forward(x)

    #plt.imshow(y_hat[0, 0, 120], cmap='')
    #plt.imshow(y_hat.cpu()[0, 1, 120])
    #plt.imshow(y_hat[0, 2, 120])
    #plt.imshow(y_hat[0, 3, 120])
    #plt.show()
    shape = list(y.size())
    shape[1] = 2
    zeros = torch.zeros(shape).cuda()

    zeros[:, 0][torch.torch.squeeze(y == 1, dim=1)] = 1
    zeros[:, 0][torch.torch.squeeze(y == 4, dim=1)] = 1
    zeros[:, 1][torch.torch.squeeze(y == 2, dim=1)] = 1

  # basic mean of all channels for now
    loss = -1*compute_per_channel_dice(y_hat, zeros)
    
    loss[loss != loss] = 0
    print('Validation loss: ')
    print(loss)
    loss = torch.sum(loss)
    self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    return loss

  def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
