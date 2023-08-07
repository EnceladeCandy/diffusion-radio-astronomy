import matplotlib.pyplot as plt
import torch
plt.style.use("dark_background")


def main(args):
    sampler = args.sampler
    sigma_likelihood = args.sigma_likelihood

    # Importing the samples
    samples = torch.load(f"../../{args.samples_folder}/idx.pt")
    img_size = int((samples.shape[-1])**0.5) # Assuming images with height = width
    samples = samples.reshape(-1, img_size, img_size)
    

    print("Creating posterior samples plot...")
    grid_size = int((args.num_samples) ** 0.5)
    if grid_size==1:
        fig = plt.figure(figsize= (8,8), dpi = 150)
        plt.imshow(samples[0].cpu(), cmap = "hot", vmin = 0, vmax = 1.48)
        plt.axis("off")
        plt.savefig(f"../../images/samples_targets/{sampler}_{sigma_likelihood:.1g}.jpg", bbox_inches = "tight", pad_inches = 0.1)

    else:
        fig, axs = plt.subplots(grid_size, grid_size, figsize = (10, 10), dpi = 150)

        k = 0
        for i in range(grid_size): 
            for j in range(grid_size): 
                axs[i, j].imshow(samples[k].cpu(), cmap = "hot")
                axs[i, j].axis("off")
                k += 1
        fig.suptitle(r"$\sigma_{lh}$ = " + f"{sigma_likelihood:.1g}", y = 0.1)
        plt.subplots_adjust(wspace = 0.1, hspace = 0.1)
        plt.savefig(f"../../images/samples_targets/{sampler}_{sigma_likelihood:.1g}.jpg", bbox_inches = "tight", pad_inches = 0.1)

if __name__ == "__main__": 
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--sigma_likelihood",        required = False,  default = 1e-4,    type = float)
    parser.add_argument("--sampler",                required = True,      default = "pc",   help = "'pc' or 'euler'")
    parser.add_argument("--num_samples",    required = True,    type=int, help = "Number of samples to view in the plot (ideally a perfect square)")
    parser.add_argument("--samples_folder",        required = False,    default ="samples_targets")
    parser.add_argument("--ms",                     required = True,    default = "HTLup_COcube")
    args = parser.parse_args()
    main(args)