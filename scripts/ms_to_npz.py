from visread import process, scatter
from casatools import msmetadata
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    fname = args.ms_dir

    # get all spws
    msmd = msmetadata()
    msmd.open(fname)
    spws = msmd.datadescids()
    msmd.done()

    uu = []
    vv = []
    weight_ls = []
    data = []
    vis_per_spw = []
    freq_per_spw = []
    sigma_rescale_spw = []
    num_chans = []

    # for a given spw
    for spw in spws:
        # calculate rescale factor from CLEAN's results
        sigma_rescale = scatter.get_sigma_rescale_datadescid(fname, spw)

        # Get visibilities
        d = process.get_processed_visibilities(fname, spw, sigma_rescale=1.0)
        
        # flatten and concatenate
        flag = d["flag"]
        chan_freq = d["frequencies"] # Hertz
        nchan = len(chan_freq)
        u = d["uu"] # meters
        v = d["vv"] # meters
        
        # Broadcasting shapes so that they are (nchan, N_vis) == flag.shape
        weight = d["weight"] 
        broadcasted_weight = weight * np.ones(shape = (nchan, weight.shape[0]))

        # Convert the uv points to klambdas given the channel frequency
        u, v = process.broadcast_and_convert_baselines(u, v, chan_freq) 
        
        # Applying the flag mask flattens each array:
        uu.append(u[~flag])
        vv.append(v[~flag])
        weight_ls.append(broadcasted_weight[~flag])
        data.append(d["data"][~flag])
        freq_per_spw.append(chan_freq)
        vis_per_spw.append(len(u[~flag]))
        sigma_rescale_spw.append(sigma_rescale)
        num_chans.append(len(chan_freq))
        
        
    np.savez(
        args.npz_dir,
        uu = np.concatenate(uu),
        vv = np.concatenate(vv),
        weight = np.concatenate(weight_ls),
        data = np.concatenate(data), 
        sigma_rescale_spw = np.array(sigma_rescale_spw),
        vis_per_spw = np.array(vis_per_spw),
        freq_per_spw = np.concatenate(freq_per_spw),
        num_freq_per_spw = np.array(num_chans)
    )

if __name__ == "__main__": 
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # Sampling parameters
    parser.add_argument("--ms_dir")
    parser.add_argument("--npz_dir")

    args = parser.parse_args()
    main(args) 


