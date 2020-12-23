import pytorch_lightning as pl
import torch

#Pytorch-lightning setup
class TumourSegmentation(pl.LightningModule):
  def __init__(self,model):
    super().__init__()
    self.model = model
  def forward(self,x):
  #  x=x.half()
    f = self.model.forward(x)

  #  print('Done forward step!')
    return f

  def training_step(self, batch, batch_idx):

    x, y = batch

    y_hat = self.model(x)
    
    # I'm not really sure why the shape is weird here, but this seems to run
    y_hat = torch.squeeze(y_hat,axis=1) 

    loss = torch.mean(torch.abs(y_hat - y))
    # this CE results in a CUDA error because the U-net implementation is strange
    #F.binary_cross_entropy(y_hat, y)
    return loss
  def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(), lr=0.02)