import os
import cv2
import mne
import h5py
import logging
import numpy as np
import pandas as pd

logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S', )
logger = logging.getLogger(__name__)

EEG_DIR = "data/24Hr Recordings"

class Dataset:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size

    def cache_train(self):
        logger.info('Creating cache file for train')
        file = h5py.File('train.h5', 'w')
        train_files = [f for f in os.listdir(EEG_DIR) if f[-3:] == 'edf']

        x_data = file.create_dataset('x_data', shape=(len(train_files),
                                     128, 128, 3), dtype=np.float16)
        y_data = file.create_dataset('y_data', shape=(len(train_files),
                                     128, 128, 1), dtype=np.float16)
        names = file.create_dataset('names', shape=(len(train_files),),
                                    dtype=h5py.special_dtype(vlen=str))

        logger.info(f'There are {len(train_files)} files in train')
        for i, fn in enumerate(train_files):
            raw_data = mne.io.read_raw_edf('data/CRFS60_1wk_withSleepStages.edf')
            df = raw_data.to_data_frame(['Activity', 'EEG', 'EMG',
                                         'Signal-Sleep'])
            df['Signal-Sleep']=df['Signal-Sleep']/1e6
            df['Signal-Sleep']=np.abs(df['Signal-Sleep']).round()
            df['Signal-Sleep'].value_counts()
            sleep_stages = []
            for i in range(8640):
                sleep_stages.append(df["Signal-Sleep"][i*5000:(i+1)*5000].mean())

            data_dict =  {'time':[],
              'eeg':[],
              'emg':[],
              'signal_str':[],
              'sleep_stage':[]}


            for i in range(6*60*24):
                data_dict['time'].append(df.time[i*5000])
                data_dict['eeg'].append(df.EEG[i*5000:(i+1)*5000].values)
                data_dict['emg'].append(df.EMG[i*5000:(i+1)*5000].values)
                data_dict['signal_str'].append(df.Activity[i*5000:(i+1)*5000].values)
                data_dict['sleep_stage'].append(df["Signal-Sleep"][i*5000])

            x_data[i, :, :, :] = img.astype(np.float16)
            mask = imread(os.path.join(MASK_TRAIN_DIR,
                                       fn.replace('.jpg', '_mask.jpg')))
            mask = mask[:, :, 0] / 255
            mask = cv2.resize(mask, (128, 128))
            mask = np.reshape(mask, (128, 128, 1))
            mask[mask >= .78] = 1
            mask[mask < .78] = 0
            y_data[i, :, :, :] = mask.astype(np.float16)
            names[i] = fn
            logger.info(f'{i+1} of {len(train_files)} files complete')
        file.close()
