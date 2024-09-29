import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataloader import SoundDataset
from linformer import Linformer
from linformer import LinformerLM
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset
import os
import random
raw_dir='/data1/xuhui/MEG_Denoise/transformer/files/'
os.environ['CUDA_VISIBLE_DEVICES'] = "4"


ds = SoundDataset()

for k in range(0, 16, 1):
    n = [0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18]
    b = np.zeros(30000)
    n.remove(k)
    a = random.sample(n, 12)

    for i in range(12):
        for j in range(2500):
            b[i*2500+j] = a[i]*2500+j
    c=np.zeros(7500)
    for i in range(12):
        n.remove(a[i])
    for i in range(3):
        for j in range(2500):
            c[i*2500+j] = n[i]*2500+j
    np.save(raw_dir+'train'+str(k)+'.npy',a)
    trds = Subset(ds, b)
    teds = Subset(ds, c)
    trds.indices = trds.indices.astype(np.int)
    teds.indices = teds.indices.astype(np.int)
    trds.indices=list(trds.indices)
    teds.indices=list(teds.indices)
    tr_dataloader = DataLoader(trds, batch_size=64)
    te_dataloader = DataLoader(teds, batch_size=64)
    # for k in range(19):
    #     n = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    #     b = np.zeros(36000)
    #     c = []
    #     d = np.zeros(45000)
    #     n.remove(k)
    #
    #     for i in range(18):
    #         for j in range(2500):
    #             d[i * 2500 + j] = n[i] * 2500 + j
    #     d = list(d)
    #     b = random.sample(d, 36000)
    #     trds = Subset(ds, b)
    #     for item in d:
    #         if item not in b:
    #             c.append(item)
    #     teds = Subset(ds, c)
    #     # trds.indices = trds.indices.astype(np.int)
    #     # teds.indices = teds.indices.astype(np.int)
    #     tr_dataloader = DataLoader(trds, batch_size=8)
    #     te_dataloader = DataLoader(teds, batch_size=8)
    class Model(pl.LightningModule):
        def __init__(self):
            super(Model, self).__init__()

            self.transformer_model = Linformer(
                dim=600,
                seq_len=306,
                depth=6,
                heads=8,
                k=2048
            )

        def forward(self, src, tgt):
            # src = src.permute(1, 0, 2)
            # tgt = tgt.permute(1, 0, 2)
            output = self.transformer_model(src)
            # output = output.permute(1, 0, 2)
            return output

        def configure_optimizers(
                self,
        ): return torch.optim.Adam(self.parameters(), lr=0.001)

        def training_step(self, batch, batch_ix):
            src, tgt = batch
            outputs = self.forward(src, tgt)
            loss = F.mse_loss(outputs, tgt)
            self.log(name='train_loss', value=loss)
            return loss

        def validation_step(self, batch, batch_ix):
            src, tgt = batch
            outputs = self.forward(src, tgt)
            loss = F.mse_loss(outputs, tgt)
            self.log(name='val_loss', value=loss, prog_bar=True)

        def test_step(self, batch, batch_ix):
            src, tgt = batch
            outputs = self.forward(src, tgt)
            self.write_prediction_dict(predictions_dict={
                'predictions': outputs,
                'actual': tgt
            })


    model = Model()
    checkpoint_callback = ModelCheckpoint(
          monitor='valid_loss',
          dirpath='checkpoint/',
          filename='epoch-{epoch:02d}-{val_loss:.2f}')
    trainer = pl.Trainer(max_epochs=100, log_every_n_steps=100, gpus=[0],
                         )

    trainer.fit(model,
                train_dataloader=tr_dataloader,
                val_dataloaders=te_dataloader)

########################################
# model = model.load_from_checkpoint('lightning_logs/version_12/checkpoints/epoch=7.ckpt')
# with torch.no_grad():
#     model.eval()
#     for batch in te_dataloader:
#         src, tgt = batch
#         pred = model(src, tgt)
#         break
#
# print(src.size())
# print(tgt.size())
# print(pred.size())
#
# sample = {
#     'src': src,
#     'tgt': tgt,
#     'pred': pred
# }
# torch.save(sample, 'predictions.pt')
# import pandas as pd
#
# sample_ix = 2
# pd.DataFrame(tgt[sample_ix].numpy()).to_csv('test_tgt.csv',index=False)
# pd.DataFrame(pred[sample_ix].numpy()).to_csv('test_pred.csv',index=False)
#
# import matplotlib.pyplot as plt
#
#
# plt.subplot(3, 1, 1)
# plt.imshow(src[sample_ix])
# plt.title('input')
#
# plt.subplot(3, 1, 2)
# plt.imshow(tgt[sample_ix])
# plt.title('actual')
#
# plt.subplot(3, 1, 3)
# plt.imshow(pred[sample_ix])
# plt.title('pred')