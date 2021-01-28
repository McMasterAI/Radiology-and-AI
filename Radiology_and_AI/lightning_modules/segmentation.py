#@title LightningModule, this is supposed to be in LightningModules but I put it here for easy modification
#Pytorch-lightning setup
import sys
sys.path.append('../')
from lib.medzoo.Unet3D import UNet3D
from lib.losses3D.basic import compute_per_channel_dice, expand_as_one_hot
import torch
import pytorch_lightning as pl

class TumourSegmentation(pl.LightningModule):
  def __init__(self, learning_rate, collator, batch_size, train_dataset, eval_dataset, in_channels=4,classes=(1,2,4)):
    super().__init__()
    self.model =  UNet3D(in_channels=in_channels, n_classes=len(classes), base_n_filter=8) #.cuda()
    self.learning_rate = learning_rate
    self.collator = collator
    self.batch_size = batch_size
    self.train_dataset = train_dataset
    self.eval_dataset = eval_dataset
    self.in_channels = in_channels
    self.classes = classes
    self.save_hyperparameters()

  def forward(self,x):
  #  x=x.half()

    f = self.model.forward(x)

  #  print('Done forward step!')
    return f

  def training_step(self, batch, batch_idx):
    x, y = batch
    x = torch.unsqueeze(x, axis=0)
    y = torch.unsqueeze(y, axis=0)

    y_hat = self.forward(x)

    shape = list(y.size())
    shape[1] = len(self.classes)
    zeros = torch.zeros(shape).cuda()

    for i in range(len(self.classes)):
      zeros[:, i][torch.squeeze(y == self.classes[i], dim=1)] = 1

      
    loss = -1*compute_per_channel_dice(y_hat, zeros)
    loss[loss != loss] = 0

  # basic mean of all channels for now
  
    for i in range(len(self.classes)):
      if self.classes[i] == 1:
        self.log('test_loss_core', loss[i], on_step=True, on_epoch=True, prog_bar=True, logger=True)
      elif self.classes[i] == 2:
        self.log('test_loss_edema', loss[i], on_step=True, on_epoch=True, prog_bar=True, logger=True)
      elif self.classes[i] == 4:
        self.log('test_loss_enhancing', loss[i], on_step=True, on_epoch=True, prog_bar=True, logger=True)
    loss = torch.sum(loss)

    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    x = torch.unsqueeze(x, axis=0)
    y = torch.unsqueeze(y, axis=0)

    y_hat = self.forward(x)

    shape = list(y.size())
    shape[1] = len(self.classes)
    zeros = torch.zeros(shape).cuda()

    for i in range(len(self.classes)):
      zeros[:, i][torch.squeeze(y == self.classes[i], dim=1)] = 1

  # basic mean of all channels for now
    loss = -1*compute_per_channel_dice(y_hat, zeros)
    loss[loss != loss] = 0
    
    for i in range(len(self.classes)):
      if self.classes[i] == 1:
        self.log('test_loss_core', loss[i], on_step=True, on_epoch=True, prog_bar=True, logger=True)
      elif self.classes[i] == 2:
        self.log('test_loss_edema', loss[i], on_step=True, on_epoch=True, prog_bar=True, logger=True)
      elif self.classes[i] == 4:
        self.log('test_loss_enhancing', loss[i], on_step=True, on_epoch=True, prog_bar=True, logger=True)
    loss = torch.sum(loss)
    return loss

    
  def train_dataloader(self):
      return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,collate_fn=self.collator)
  def val_dataloader(self):
      return torch.utils.data.DataLoader(self.eval_dataset, batch_size=self.batch_size,collate_fn=self.collator)          

  def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
