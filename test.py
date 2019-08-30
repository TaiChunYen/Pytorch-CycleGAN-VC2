import os
import numpy as np
import argparse
import torch
import time
import librosa
import pickle

import preprocess
from trainingDataset import trainingDataset
from model_VC2 import Generator, Discriminator

class CycleGANTest:
    def __init__(self,
                 logf0s_normalization,
                 mcep_normalization,
                 model_checkpoint,
                 validation_A_dir,
                 output_A_dir):

        logf0s_normalization = np.load(logf0s_normalization)
        self.log_f0s_mean_A = logf0s_normalization['mean_A']
        self.log_f0s_std_A = logf0s_normalization['std_A']
        self.log_f0s_mean_B = logf0s_normalization['mean_B']
        self.log_f0s_std_B = logf0s_normalization['std_B']

        mcep_normalization = np.load(mcep_normalization)
        self.coded_sps_A_mean = mcep_normalization['mean_A']
        self.coded_sps_A_std = mcep_normalization['std_A']
        self.coded_sps_B_mean = mcep_normalization['mean_B']
        self.coded_sps_B_std = mcep_normalization['std_B']

        self.validation_A_dir = validation_A_dir
        self.output_A_dir = output_A_dir

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.generator_A2B = Generator().to(self.device)
        self.generator_A2B.eval()

        checkPoint = torch.load(model_checkpoint)
        self.generator_A2B.load_state_dict(
            state_dict=checkPoint['model_genA2B_state_dict'])
        print("load model checkpoint finish!")


    def validation_for_A_dir(self):
        num_mcep = 24
        sampling_rate = 16000
        frame_period = 5.0
        n_frames = 128
        validation_A_dir = self.validation_A_dir
        output_A_dir = self.output_A_dir

        print("Generating Test Data B from A...")
        for file in os.listdir(validation_A_dir):
            filePath = os.path.join(validation_A_dir, file)
            wav, _ = librosa.load(filePath, sr=sampling_rate, mono=True)
            wav = preprocess.wav_padding(wav=wav,
                                         sr=sampling_rate,
                                         frame_period=frame_period,
                                         multiple=4)
            f0, timeaxis, sp, ap = preprocess.world_decompose(
                wav=wav, fs=sampling_rate, frame_period=frame_period)
            f0_converted = preprocess.pitch_conversion(f0=f0,
                                                       mean_log_src=self.log_f0s_mean_A,
                                                       std_log_src=self.log_f0s_std_A,
                                                       mean_log_target=self.log_f0s_mean_B,
                                                       std_log_target=self.log_f0s_std_B)
            coded_sp = preprocess.world_encode_spectral_envelop(
                sp=sp, fs=sampling_rate, dim=num_mcep)
            coded_sp_transposed = coded_sp.T
            coded_sp_norm = (coded_sp_transposed -
                             self.coded_sps_A_mean) / self.coded_sps_A_std
            coded_sp_norm = np.array([coded_sp_norm])

            if torch.cuda.is_available():
                coded_sp_norm = torch.from_numpy(coded_sp_norm).cuda().float()
            else:
                coded_sp_norm = torch.from_numpy(coded_sp_norm).float()

            coded_sp_converted_norm = self.generator_A2B(coded_sp_norm)
            coded_sp_converted_norm = coded_sp_converted_norm.cpu().detach().numpy()
            coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)
            coded_sp_converted = coded_sp_converted_norm * \
                self.coded_sps_B_std + self.coded_sps_B_mean
            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            decoded_sp_converted = preprocess.world_decode_spectral_envelop(
                coded_sp=coded_sp_converted, fs=sampling_rate)
            wav_transformed = preprocess.world_speech_synthesis(f0=f0_converted,
                                                                decoded_sp=decoded_sp_converted,
                                                                ap=ap,
                                                                fs=sampling_rate,
                                                                frame_period=frame_period)
            librosa.output.write_wav(path=os.path.join(output_A_dir, os.path.basename(file)),
                                     y=wav_transformed,
                                     sr=sampling_rate)
        print("finish!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Test CycleGAN")

    logf0s_normalization_default = '../cache/logf0s_normalization.npz'
    mcep_normalization_default = '../cache/mcep_normalization.npz'
    coded_sps_A_norm = '../cache/coded_sps_A_norm.pickle'
    coded_sps_B_norm = '../cache/coded_sps_B_norm.pickle'
    model_checkpoint = '../cache/model_checkpoint/'
    resume_training_at = '../cache/model_checkpoint/_CycleGAN_CheckPoint'
    resume_training_at = None

    validation_A_dir_default = '../data/vcc2016_training/evaluation_all/SF1/'
    output_A_dir_default = '../data/vcc2016_training/converted_sound/SF1'

    validation_B_dir_default = '../data/vcc2016_training/evaluation_all/TF2/'
    output_B_dir_default = '../data/vcc2016_training/converted_sound/TF2/'

    parser.add_argument('--logf0s_normalization', type=str,
                        help="Cached location for log f0s normalized", default=logf0s_normalization_default)
    parser.add_argument('--mcep_normalization', type=str,
                        help="Cached location for mcep normalization", default=mcep_normalization_default)
    parser.add_argument('--model_checkpoint', type=str,
                        help="location where your model", default=model_checkpoint)
    parser.add_argument('--test_A_dir', type=str,
                        help="test set for sound source A", default=validation_A_dir_default)
    parser.add_argument('--output_A_dir', type=str,
                        help="output for converted Sound Source A", default=output_A_dir_default)

    argv = parser.parse_args()

    logf0s_normalization = argv.logf0s_normalization
    mcep_normalization = argv.mcep_normalization
    model_checkpoint = argv.model_checkpoint

    validation_A_dir = argv.test_A_dir
    output_A_dir = argv.output_A_dir

    cycleGAN = CycleGANTest(logf0s_normalization=logf0s_normalization,
                                mcep_normalization=mcep_normalization,
                                model_checkpoint=model_checkpoint,
                                validation_A_dir=validation_A_dir,
                                output_A_dir=output_A_dir)


    cycleGAN.validation_for_A_dir()



        
