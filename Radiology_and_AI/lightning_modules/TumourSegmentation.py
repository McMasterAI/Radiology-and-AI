import torch
import pytorch_lightning as pl
import sys
sys.path.append('../MedicalZooPytorch')
from lib.medzoo.Unet3D import UNet3D
from lib.losses3D.basic import compute_per_channel_dice, expand_as_one_hot
from torch.utils.data import Dataset, DataLoader, random_split
class TumourSegmentation(pl.LightningModule):
    def __init__(self, train_dataset, val_dataset, col_fn, learning_rate, num_loading_cpus=1, batch_size=1,in_channels=4,classes=(1,2,4)):
        super().__init__()    
        self.model =  UNet3D(in_channels=in_channels, n_classes=len(classes), base_n_filter=8)    
        self.learning_rate = learning_rate
        self.in_channels = in_channels
        self.num_loading_cpus = num_loading_cpus
        self.batch_size = batch_size
        self.classes = classes
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.col_fn = col_fn
        self.normalization = torch.nn.Sigmoid()        

    def forward(self,x):
        f = self.model.forward(x)
        return f

    def training_step(self, batch, batch_idx):
        x= batch['data']
        y = torch.cat([batch['seg'][:,x].unsqueeze(dim=1) for x in self.classes],dim = 1)
        y_hat = self.forward(x)
        y_hat = self.normalization(y_hat)
        loss = -1*compute_per_channel_dice(y_hat, y)
        loss[loss != loss] = 0

        for i in range(len(self.classes)):    
            if self.classes[i] == 1:
                self.log('train_loss_core',loss[i],prog_bar=True,logger=True)
            elif self.classes[i] == 2:
                self.log('train_loss_edema',loss[i],prog_bar=True,logger=True)
            elif self.classes[i] == 4:
                self.log('train_loss_enhancing',loss[i],prog_bar=True,logger=True)
        loss = torch.mean(loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x= batch['data']
        y = torch.cat([batch['seg'][:,1:3],batch['seg'][:,4].unsqueeze(dim=1)],dim = 1)
        y_hat = self.forward(x)
        y_hat = self.normalization(y_hat)

        loss = -1*compute_per_channel_dice(y_hat, y)
        loss[loss != loss] = 0
        for i in range(len(self.classes)):      
            if self.classes[i] == 1:
                self.log('test_loss_core',loss[i],prog_bar=True,logger=True)
            elif self.classes[i] == 2:
                self.log('test_loss_edema',loss[i],prog_bar=True,logger=True)
            elif self.classes[i] == 4:
                self.log('test_loss_enhancing',loss[i],prog_bar=True,logger=True)
        loss = torch.mean(loss)
        return loss
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,batch_size=self.batch_size,num_workers=self.num_loading_cpus,collate_fn=self.col_fn)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,batch_size=self.batch_size,num_workers=self.num_loading_cpus,collate_fn=self.col_fn)     

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
