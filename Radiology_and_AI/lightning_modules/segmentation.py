from lib.medzoo.Unet3D import UNet3D
from lib.losses3D.basic import compute_per_channel_dice, expand_as_one_hot
import torch
import numpy
from visuals.display_functions import display_brain_and_segs
import pytorch_lightning as pl

class TumourSegmentation(pl.LightningModule):
  def __init__(self, learning_rate, train_collator, val_collator, batch_size, train_dataset, eval_dataset, in_channels=4,classes=(1,2,4),display_seg=False):
    super().__init__()
    self.model =  UNet3D(in_channels=in_channels, n_classes=len(classes), base_n_filter=8) #.cuda()
    self.learning_rate = learning_rate
    self.train_collator = train_collator
    self.val_collator = val_collator
    self.batch_size = batch_size
    self.train_dataset = train_dataset
    self.eval_dataset = eval_dataset
    self.in_channels = in_channels
    self.classes = classes
    self.display_seg = display_seg
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
        self.log('train_loss_core',loss[i],prog_bar=True,logger=True)
      elif self.classes[i] == 2:
        self.log('train_loss_edema',loss[i],prog_bar=True,logger=True)
      elif self.classes[i] == 4:
        self.log('train_loss_enhancing',loss[i],prog_bar=True,logger=True)
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
        self.log('test_loss_core',loss[i],prog_bar=True,logger=True)
      elif self.classes[i] == 2:
        self.log('test_loss_edema',loss[i],prog_bar=True,logger=True)
      elif self.classes[i] == 4:
        self.log('test_loss_enhancing',loss[i],prog_bar=True,logger=True)
    loss = torch.sum(loss)

    seg = y.detach().cpu().numpy()
    #print(y_hat)

    y_hat_disp = torch.zeros(shape).cuda()    
    y_hat_disp[torch.squeeze(y_hat >= 0.6, dim=1)] = 1
    y_hat_disp = y_hat_disp.squeeze()
    our_seg = y_hat_disp.detach().cpu().numpy()


    #remove the extra dimension in the segmentation
    seg = numpy.squeeze(seg)
    #Creating boolean arrays for each segmentation type
    seg_all = seg != 0
    seg_1 = seg == 1
    seg_2 = seg == 2
    seg_4 = seg == 4
    if self.display_seg:
      display_brain_and_segs(our_seg[0,:],downsize_factor=5,fig_size=(5,5))
      display_brain_and_segs(seg_1,downsize_factor=5,fig_size=(5,5))

    return loss

    
  def train_dataloader(self):
      return torch.utils.data.DataLoader(self.train_dataset,batch_size=self.batch_size,collate_fn=self.train_collator)
  def val_dataloader(self):
      return torch.utils.data.DataLoader(self.eval_dataset,batch_size=self.batch_size,collate_fn=self.val_collator)          

  def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(), lr=self.learning_rate)