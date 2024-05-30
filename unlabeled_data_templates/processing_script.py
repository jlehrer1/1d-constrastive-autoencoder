import braingeneers.analysis
#import numpy as np
import pandas as pd
import braingeneers.utils.smart_open_braingeneers as smart_open
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import os
import braingeneers as bg
import argparse

bg.set_default_endpoint('/Users/jesusgf/Desktop/ephys_autoencoder/1d-constrastive-autoencoder/unlabeled_data_templates/')



argparser = argparse.ArgumentParser()
argparser.add_argument('--uuid', type=str, required=True)
argparser.add_argument('-r','--round', type=str, required=True)
args = argparser.parse_args()


try:
    phy = braingeneers.analysis.load_spike_data(uuid = args.uuid)
    trains = phy.train
    all_isi = phy.interspike_intervals()

    waveforms = []
    isi_dist = []
    bad_indices = []
    for idx, isi in enumerate(all_isi):
        hist, _ = np.histogram(isi[isi < 100], bins=100, density=True) #This is in miliseconds# Tal had (bin widths were set to 1/15th of the median ISI for a given unit)
        if not np.isnan(hist).all():
            isi_dist.append(hist)
        else:
            bad_indices.append(idx)

    templates = [phy.neuron_attributes[i].template for i in range(len(phy.neuron_attributes))]
    before, after = 20, 30
    # dir = "2023-09-16-efi-mouse-5plex-official/19894_cleaned"
    cut_data = np.empty((0, 50))
    for i in range(len(phy.neuron_attributes)):
        d = templates[i]
        # find the peak of the spike
        d = d - np.mean(d[:before])  # subtract the baseline
        d = d / np.max(d)  # normalize
        if abs(np.min(d)) > abs(np.max(d)):  # this is a negative spike
            peak_index = np.argmin(d)
        else:
            peak_index = np.argmax(d)  # this is a positive spike
        if peak_index >= before and peak_index < 100-after:
            # save the cut template
            new_spike = d[peak_index-before: peak_index+after]
            cut_data = np.vstack((cut_data, new_spike))

    waveform_data = np.delete(cut_data, np.array(bad_indices), axis=0)

    waveforms.append(waveform_data)

    #Convert list to numpy array
    waveforms = np.array(waveforms)
    isi_dist = np.array(isi_dist)

    #Save the data
    np.save(f'./processed_data/{args.uuid}{args.round}-templates.npy', waveforms)
    np.save(f'./processed_data/{args.uuid}{args.round}-isi_distribution.npy', isi_dist)

except Exception as e:
    print(e)
    print(f"Error processing {args.uuid}")