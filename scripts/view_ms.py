import matplotlib.pyplot as plt
from utils import fits_to_tensor

# Creates a plot to view the dirty image and the psf of a measurement set 

def main(args): 
    path = f"../../data_targets2/{args.ms}"
    
    psf = fits_to_tensor(path + "_psf.fits")
    dirty_image = fits_to_tensor(path + ".fits")

    print("Images of size", dirty_image.shape)
    plt.style.use('dark_background')
    fig, axs = plt.subplots(1, 2, figsize = (7, 3.5), dpi = 150)
    cmap = args.cmap

    for i in range(len(axs)):
        axs[i].axis("off")

    
    im = axs[0].imshow(psf, cmap = cmap)
    axs[0].set_title("Psf")
    plt.colorbar(im, fraction = 0.046)

    
    #plt.imshow(dirty_image, cmap, norm=plt.cm.colors.LogNorm(vmin=0, clip=True))
    #plt.title("Dirty image")
    
    im = axs[1].imshow(dirty_image, cmap = cmap, origin = "lower")
    axs[1].set_title("Dirty image")
    plt.colorbar(im, fraction = 0.046)
    plt.subplots_adjust(wspace = 0.2)
    plt.savefig(f"../../images/dirty_images/{args.ms}.jpeg", bbox_inches = "tight", pad_inches = 0.1)


if __name__ == "__main__": 
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Path to the measurement set:
    parser.add_argument("--ms",     required = True,    help = "Name of the measurement set")
    parser.add_argument("--cmap",   required = False,   default = "magma")
    
    args = parser.parse_args()
    main(args)
