import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from dataloader import SoundDataset
from linformer import Linformer
import numpy as np
# from scipy.io import savemat
import scipy.io as io
import pandas as pd
import os
from pytorch_lightning.callbacks import ModelCheckpoint


# ttl_nums = len(ds)
# split_rate = 1
# test_nums = int(ttl_nums * split_rate)
# trds, teds = random_split(ds, [ttl_nums-test_nums, test_nums])
# tr_dataloader = DataLoader(trds, batch_size=8)
# te_dataloader = DataLoader(teds, batch_size=8)
# test_dataloader = DataLoader(teds, batch_size=2500)


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
    ):return torch.optim.Adam(self.parameters(), lr=0.001)


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
            'predictions':outputs,
            'actual':tgt
        })

ds = SoundDataset()
# n = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
for i in range(6,7,1):
    a=i
    b = np.zeros(2500)
    for k in range(1):
        for j in range(2500):
            b[k*2500+j] = a*2500+j
    teds = Subset(ds, b)
    teds.indices = teds.indices.astype(np.int)
    teds.indices=list(teds.indices)
    test_dataloader = DataLoader(teds, batch_size=2500)
    model = Model()
    c=237
    filepath = '/data1/xuhui/MEG_Denoise/transformer/files/realdata/lightning_logs/version_'+str(c)+'/checkpoints/'
    f = os.listdir(filepath)
    for k in f:
        print(k)
    model = model.load_from_checkpoint(filepath+str(k))


    with torch.no_grad():
        model.eval()
        for batch in test_dataloader:
            src, tgt = batch
            pred = model(src, tgt)
            break


    print(src.size())
    print(tgt.size())
    print(pred.size())

    sample = {
        'src': src,
        'tgt': tgt,
        'pred': pred
    }
    torch.save(sample, 'predictions.pt')

    sample_ix=10
# pd.DataFrame(tgt.numpy()).to_csv('test_tgt.csv',index=False)
# pd.DataFrame(pred.numpy()).to_csv('test_pred.csv',index=False)
# pd.DataFrame(src.numpy()).to_csv('test_src.csv',index=False)

    import matplotlib.pyplot as plt
#
#
    plt.subplot(3, 1, 1)
    plt.plot(src[sample_ix].T)
    plt.xlabel('Time(ms)')
    plt.ylabel('Amplitude')
    plt.title('Input')

    plt.subplot(3, 1, 2)
    plt.plot(tgt[sample_ix].T)
    plt.xlabel('Time(ms)')
    plt.ylabel('Amplitude')
    plt.title('Output')

    plt.subplot(3, 1, 3)
    plt.plot(pred[sample_ix].T)
    plt.xlabel('Time(ms)')
    plt.ylabel('Amplitude')
    plt.title('Pred')

    plt.tight_layout(h_pad=1.0)
    plt.savefig('result_version'+str(i)+'_100.png')

# for i in range(10):
#     sample_ix=i
#     plt.subplot(3, 1, 1)
#     plt.plot(src[sample_ix].T)
#     plt.xlabel('Time(ms)')
#     plt.ylabel('Amplitude')
#     plt.title('Input')
#
#     plt.subplot(3, 1, 2)
#     plt.plot(tgt[sample_ix].T)
#     plt.xlabel('Time(ms)')
#     plt.ylabel('Amplitude')
#     plt.title('Output')
#
#     plt.subplot(3, 1, 3)
#     plt.plot(pred[sample_ix].T)
#     plt.xlabel('Time(ms)')
#     plt.ylabel('Amplitude')
#     plt.title('Pred')
#pytho
#     plt.tight_layout(h_pad=1.0)
#     plt.savefig('result_'+str(i)+'.png')
# #
    np.save(file='input_sim_version'+str(i)+'_20.npy', arr=src)
    np.save(file='output_sim_version'+str(i)+'_20.npy', arr=tgt)
    np.save(file='pred_sim_version'+str(i)+'_20.npy', arr=pred)
#
    input_ori=np.load('input_sim_version'+str(i)+'_20.npy')
    output_ori=np.load('output_sim_version'+str(i)+'_20.npy')
    pred_ori=np.load('pred_sim_version'+str(i)+'_20.npy')

    io.savemat('input_sim_version'+str(i)+'_20.mat', {"input_ori": input_ori})
    io.savemat('output_sim_version'+str(i)+'_20.mat', {"output_ori": output_ori})
    io.savemat('pred_sim_version'+str(i)+'_20.mat', {"pred_ori": pred_ori})