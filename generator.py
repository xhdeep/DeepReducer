import os
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import pandas as pd
from copy import deepcopy
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# work_dir = '/home/xuhui/MEG-Denoise/MEG_denoise/logs/'
raw_dir = '/data1/xuhui/MEG_Denoise/transformer/files/'
os.makedirs('data_inputs', exist_ok=True)
os.makedirs('data_outputs', exist_ok=True)

limited_num = 20

output_datas1= np.load(raw_dir+'output_sim_sub12_20_2500.npy')
input_datas1= np.load(raw_dir+'input_sim_sub12_20_2500.npy')
# output_datas1 = output_datas1[2000:,:,:]
# input_datas1 = input_datas1[2500:,:,:]

output_datas2= np.load(raw_dir+'output_sim_sub13_20_2500.npy')
input_datas2= np.load(raw_dir+'input_sim_sub13_20_2500.npy')
# output_datas2 = output_datas2[2500:,:,:]
# input_datas2 = input_datas2[2500:,:,:]

output_datas3= np.load(raw_dir+'output_sim_sub14_20_2500.npy')
input_datas3= np.load(raw_dir+'input_sim_sub14_20_2500.npy')
# output_datas3 = output_datas3[2500:,:,:]
# input_datas3 = input_datas3[2500:,:,:]

output_datas4= np.load(raw_dir+'output_sim_sub15_20_2500.npy')
input_datas4= np.load(raw_dir+'input_sim_sub15_20_2500.npy')
# output_datas4 = output_datas4[2500:,:,:]
# input_datas4 = input_datas4[2500:,:,:]

output_datas5= np.load(raw_dir+'output_sim_sub16_20_2500.npy')
input_datas5= np.load(raw_dir+'input_sim_sub16_20_2500.npy')
# output_datas5 = output_datas5[2500:,:,:]
# input_datas5 = input_datas5[:2000,:,:]

output_datas6= np.load(raw_dir+'output_sim_sub18_20_2500.npy')
input_datas6= np.load(raw_dir+'input_sim_sub18_20_2500.npy')
# output_datas6 = output_datas6[:2000,:,:]
# input_datas6 = input_datas6[:2000,:,:]

output_datas7= np.load(raw_dir+'output_sim_sub20_20_2500.npy')
input_datas7= np.load(raw_dir+'input_sim_sub20_20_2500.npy')
# output_datas7 = output_datas7[:2000,:,:]
# input_datas7 = input_datas7[:2000,:,:]

output_datas8= np.load(raw_dir+'output_sim_sub24_20_2500.npy')
input_datas8= np.load(raw_dir+'input_sim_sub24_20_2500.npy')
# output_datas8 = output_datas8[:2000,:,:]
# input_datas8 = input_datas8[:2000,:,:]
#
output_datas9= np.load(raw_dir+'output_sim_sub25_20_2500.npy')
input_datas9= np.load(raw_dir+'input_sim_sub25_20_2500.npy')
# output_datas9 = output_datas9[:2000,:,:]
# input_datas9 = input_datas9[:2000,:,:]

output_datas10= np.load(raw_dir+'output_sim_sub27_20_2500.npy')
input_datas10= np.load(raw_dir+'input_sim_sub27_20_2500.npy')
# output_datas10 = output_datas10[:2000,:,:]
# input_datas10 = input_datas10[:2000,:,:]

output_datas11= np.load(raw_dir+'output_sim_sub34_20_2500.npy')
input_datas11= np.load(raw_dir+'input_sim_sub34_20_2500.npy')
# output_datas11 = output_datas11[:2000,:,:]
# input_datas11 = input_datas11[:2000,:,:]

output_datas12= np.load(raw_dir+'output_sim_sub35_20_2500.npy')
input_datas12= np.load(raw_dir+'input_sim_sub35_20_2500.npy')
# output_datas12 = output_datas12[:2000,:,:]
# input_datas12 = input_datas12[:2000,:,:]

output_datas13= np.load(raw_dir+'output_sim_sub37_20_2500.npy')
input_datas13= np.load(raw_dir+'input_sim_sub37_20_2500.npy')
# output_datas13 = output_datas13[:2000,:,:]
# input_datas13 = input_datas13[:2000,:,:]

output_datas14= np.load(raw_dir+'output_sim_sound_20_2500.npy')
input_datas14= np.load(raw_dir+'input_sim_sound_20_2500.npy')
# output_datas14 = output_datas14[:2000,:,:]
# input_datas14 = input_datas14[:2000,:,:]
#
# output_datas15= np.load(raw_dir+'sub37_dataset_'+str(limited_num)+'_output_2500.npy')
# input_datas15= np.load(raw_dir+'sub37_dataset_'+str(limited_num)+'_input_2500.npy')
# output_datas15 = output_datas15[:2500,:,:]
# input_datas15 = input_datas15[:2500,:,:]
#
# output_datas16= np.load(raw_dir+'sound_dataset_'+str(limited_num)+'_output_2500.npy')
# input_datas16= np.load(raw_dir+'sound_dataset_'+str(limited_num)+'_input_2500.npy')
# output_datas16 = output_datas16[:2500,:,:]
# input_datas16 = input_datas16[:2500,:,:]
#
# output_datas17= np.load(raw_dir+'sub27_dataset_'+str(limited_num)+'_output_2500.npy')
# input_datas17= np.load(raw_dir+'sub27_dataset_'+str(limited_num)+'_input_2500.npy')
# output_datas17 = output_datas17[:2500,:,:]
# input_datas17 = input_datas17[:2500,:,:]
#
# output_datas18= np.load(raw_dir+'sub36_dataset_'+str(limited_num)+'_output_2500.npy')
# input_datas18= np.load(raw_dir+'sub36_dataset_'+str(limited_num)+'_input_2500.npy')
# output_datas18 = output_datas18[:2500,:,:]
# input_datas18 = input_datas18[:2500,:,:]

# output_datas19= np.load(raw_dir+'sub33_dataset_'+str(limited_num)+'_output_2500.npy')
# input_datas19= np.load(raw_dir+'sub33_dataset_'+str(limited_num)+'_input_2500.npy')
# output_datas19 = output_datas18[:500,:,:]
# input_datas19 = input_datas18[:500,:,:]
#
# output_datas20= np.load(raw_dir+'sub20_dataset_'+str(limited_num)+'_output_2500.npy')
# input_datas20= np.load(raw_dir+'sub20_dataset_'+str(limited_num)+'_input_2500.npy')
# output_datas20 = output_datas18[:500,:,:]
# input_datas20 = input_datas18[:500,:,:]
#
# output_datas21= np.load(raw_dir+'sub32_dataset_'+str(limited_num)+'_output_2500.npy')
# input_datas21= np.load(raw_dir+'sub32_dataset_'+str(limited_num)+'_input_2500.npy')
# output_datas21 = output_datas18[:500,:,:]
# input_datas21 = input_datas18[:500,:,:]
#
# output_datas22= np.load(raw_dir+'sub22_dataset_'+str(limited_num)+'_output_2500.npy')
# input_datas22= np.load(raw_dir+'sub22_dataset_'+str(limited_num)+'_input_2500.npy')
# output_datas22 = output_datas18[:500,:,:]
# input_datas22 = input_datas18[:500,:,:]
#
# output_datas23= np.load(raw_dir+'sub30_dataset_'+str(limited_num)+'_output_2500.npy')
# input_datas23= np.load(raw_dir+'sub30_dataset_'+str(limited_num)+'_input_2500.npy')
# output_datas23 = output_datas18[:500,:,:]
# input_datas23 = input_datas18[:500,:,:]
#
# output_datas24= np.load(raw_dir+'sub24_dataset_'+str(limited_num)+'_output_2500.npy')
# input_datas24= np.load(raw_dir+'sub24_dataset_'+str(limited_num)+'_input_2500.npy')
# output_datas24 = output_datas18[:500,:,:]
# input_datas24 = input_datas18[:500,:,:]
#
# output_datas25= np.load(raw_dir+'sub25_dataset_'+str(limited_num)+'_output_2500.npy')
# input_datas25= np.load(raw_dir+'sub25_dataset_'+str(limited_num)+'_input_2500.npy')
# output_datas25 = output_datas18[:500,:,:]
# input_datas25 = input_datas18[:500,:,:]
#
# output_datas26= np.load(raw_dir+'sub29_dataset_'+str(limited_num)+'_output_2500.npy')
# input_datas26= np.load(raw_dir+'sub29_dataset_'+str(limited_num)+'_input_2500.npy')
# output_datas26 = output_datas18[:500,:,:]
# input_datas26 = input_datas18[:500,:,:]
#
# output_datas27= np.load(raw_dir+'sub27_dataset_'+str(limited_num)+'_output_2500.npy')
# input_datas27= np.load(raw_dir+'sub27_dataset_'+str(limited_num)+'_input_2500.npy')
# output_datas27 = output_datas18[:500,:,:]
# input_datas27 = input_datas18[:500,:,:]
#
# output_datas28= np.load(raw_dir+'sub28_dataset_'+str(limited_num)+'_output_2500.npy')
# input_datas28= np.load(raw_dir+'sub28_dataset_'+str(limited_num)+'_input_2500.npy')
# output_datas28 = output_datas18[:500,:,:]
# input_datas28 = input_datas18[:500,:,:]


# output_datas23= np.load(raw_dir+'sub33_dataset_'+str(limited_num)+'_output_2500.npy')
# input_datas23= np.load(raw_dir+'sub33_dataset_'+str(limited_num)+'_input_2500.npy')
output_datas = np.concatenate((output_datas1,output_datas2,output_datas3,output_datas4,output_datas5,output_datas6,output_datas7,output_datas8,output_datas9,output_datas10,output_datas11,output_datas12,output_datas13,output_datas14),axis=0)
input_datas = np.concatenate((input_datas1,input_datas2,input_datas3,input_datas4,input_datas5,input_datas6,input_datas7,input_datas8,input_datas9,input_datas10,input_datas11,input_datas12,input_datas13,input_datas14),axis=0)




#(n_example,306,600) --> (n_example,306,600,1)
# X = np.expand_dims(input_datas_norm,axis=3)
# y = np.expand_dims(output_datas_norm,axis=3)

# X = np.expand_dims(input_datas[:,:,0:600],axis=3)  #(n_example,306,600) --> (n_example,306,600,1)
# y = np.expand_dims(output_datas[:,:,0:600],axis=3)
# logging.info("X shape:{}---y shape:{}".format(X.shape,y.shape))
# import sys
# sys.exit()
data_list_inputs = []
root = 'data_inputs'
# data = input_datas_norm['input_datas']
input_datas_mean = np.mean(input_datas[:,:,0:600])
input_datas_std = np.std(input_datas[:,:,0:600])
input_datas_norm = deepcopy(input_datas[:,:,0:600])
input_datas_norm = (input_datas[:,:,0:600]-input_datas_mean)/input_datas_std
input_datas_norm = input_datas_norm.astype(np.float32)
for ids, i in tqdm(enumerate(input_datas_norm)):
    save_ph = os.path.join(root, str(ids)+'.npy')
    data_list_inputs.append(save_ph)
    np.save(save_ph, i)

data_list_outputs = []
root = 'data_outputs'
# data = output_datas_norm['output_datas']
output_datas_mean = np.mean(output_datas[:,:,0:600])
output_datas_std = np.std(output_datas[:,:,0:600])
output_datas_norm = deepcopy(output_datas[:,:,0:600])
output_datas_norm = (output_datas[:,:,0:600]-output_datas_mean)/output_datas_std
output_datas_norm = output_datas_norm.astype(np.float32)
# (2500, 306, 601)
for ids, i in tqdm(enumerate(output_datas_norm)):
    save_ph = os.path.join(root, str(ids)+'.npy')
    data_list_outputs.append(save_ph)
    np.save(save_ph, i)

df = pd.DataFrame()
df['input'] = data_list_inputs
df['output'] = data_list_outputs
df.to_csv('train_path.csv', index=False)

