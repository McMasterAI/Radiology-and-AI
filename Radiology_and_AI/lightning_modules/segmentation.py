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
    shape[1] = 3
    zeros = torch.zeros(shape).cuda()

    for i in range(1, 4):
      zeros[:, i-1][torch.squeeze(y == i, dim=1)] = 1

  # basic mean of all channels for now
    loss = compute_per_channel_dice(y_hat, zeros)
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
    shape[1] = 3
    zeros = torch.zeros(shape).cuda()

    for i in range(1, 4):
      zeros[:, i-1][torch.squeeze(y == i, dim=1)] = 1

  # basic mean of all channels for now
    loss = compute_per_channel_dice(y_hat, zeros)
    loss[loss != loss] = 0
    print('Validation loss: ')
    print(loss)
    loss = torch.sum(loss)
    self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    return loss

  def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(), lr=0.02)
