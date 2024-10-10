import os
import shutil

import numpy as np
from tqdm import tqdm

from utils import utils


def create_sets(wav_files : np.ndarray, 
                background_noises: np.ndarray, 
                impulse_responses: np.ndarray,
                train_ratio,
                rng,
                ):

    # Split wav files, background noises and impulse responses into train/val sets
    train_set, val_set = utils.split_to_train_val_sets(wav_files, train_ratio, rng)
    train_test_info = (  f"Total files: {len(wav_files)}\n"
                       + f"Train files: {train_set.size}\n"
                       + f"Test files: {val_set.size}")
    print(train_test_info)

    train_impulse_responses, val_impulse_responses = \
        utils.split_to_train_val_sets(impulse_responses, train_ratio, rng)
    print(
        f"Train impulse responses: {train_impulse_responses.size}"
        + f"\nTest impulse responses: {val_impulse_responses.size}"
    )

    train_background_noises, val_background_noises = \
        utils.split_to_train_val_sets(background_noises, train_ratio, rng)
    print(
        f"Train background noises: {train_background_noises.size}"
        + f"\nTest background noises: {val_background_noises.size}"
    ) 
    
    current_dir = os.getcwd()
    tmp_folder = os.path.join(current_dir, 'online_data')
    
    train_subfolders = ['train_set', 'train_impulse_responses', 'train_background_noises']
    val_subfolders = ['val_set', 'val_impulse_responses', 'val_background_noises']

    # Split wav files to the corresponding folders
    try:
        os.mkdir(tmp_folder)
    except (FileExistsError, FileNotFoundError) as err:
        print(f"Exception occured: {err}")
        raise err
    
    try:        
        train_folder = os.path.join(tmp_folder, 'train')
        val_folder = os.path.join(tmp_folder, 'val')
        
        os.mkdir(train_folder)
        os.mkdir(val_folder)
        
        for subfolder in train_subfolders:
            os.mkdir(os.path.join(train_folder, subfolder))
        for subfolder in val_subfolders:
            os.mkdir(os.path.join(val_folder, subfolder))
    except (FileExistsError, FileNotFoundError) as err:
        print(f"Exception occured: {err}")
        raise err
    
    
    for wav in tqdm(train_set, desc='Copying training files'):
        shutil.copy(wav, train_folder+'/'+train_subfolders[0])
    for wav in tqdm(train_impulse_responses, desc='Copying training impulse responses'):
        shutil.copy(wav, train_folder+'/'+train_subfolders[1])
    for wav in tqdm(train_background_noises, desc='Copying training background noises'):
        shutil.copy(wav, train_folder+'/'+train_subfolders[2])

    for wav in tqdm(val_set, desc='Copying validation files'):
        shutil.copy(wav, val_folder+'/'+val_subfolders[0])
    for wav in tqdm(val_impulse_responses, desc='Copying validation impulse responses'):
        shutil.copy(wav, val_folder+'/'+val_subfolders[1])
    for wav in tqdm(val_background_noises, desc='Copying validation background noises'):
        shutil.copy(wav, val_folder+'/'+val_subfolders[2])
    
    # return train_set, val_set, train_impulse_responses, val_impulse_responses, \
    #         train_background_noises, val_background_noises

    train_path = train_folder + '/' + train_subfolders[0]
    train_impulse_responses_path = train_folder + '/' + train_subfolders[1]
    train_background_noises_path = train_folder + '/' + train_subfolders[2]
    val_path = val_folder + '/' + val_subfolders[0]
    val_impulse_responses_path = val_folder + '/' + val_subfolders[1]
    val_background_noises_path = val_folder + '/' + val_subfolders[2]
    
    return train_path, train_impulse_responses_path, train_background_noises_path, \
        val_path, val_impulse_responses_path, val_background_noises_path



