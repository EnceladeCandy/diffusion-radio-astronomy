from casatools import table, ms
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
import quick_imaging
import re
import os


def overwrite_visibility(ms_path):
    # Create a table and ms tool
    tb_tool = table()
    ms_tool = ms()

    # Open the new MS
    tb_tool.open(ms_path, nomodify=False)
    ms_tool.open(ms_path)

    # Get the existing spectral windows
    # if ... # Make this a conditional SPW call. Otherwise replace with None and skip the spw indexing.
    spw_info = ms_tool.getspectralwindowinfo()
    num_spws = len(spw_info)

    # Close the ms tool
    ms_tool.close()

    # Iterate over spectral windows to read and write data.
    for spw in range(num_spws):
        # Select the rows for this spectral window
        tb_spw = tb_tool.query(f"DATA_DESC_ID=={spw}")

        # Get the existing visibilities
        existing_visibilities = tb_spw.getcol("DATA")
        std_dev = tb_spw.getcol("WEIGHT")[:, np.newaxis] ** -0.5
        mean = np.zeros(existing_visibilities.shape)

        # Generate data
        real = np.random.normal(mean, std_dev)
        imaginary = 1j * np.random.normal(mean, std_dev)
        new_visibilities = real + imaginary

        if existing_visibilities.shape != new_visibilities.shape:
            raise ValueError(
                f"Shape of new visibilities {new_visibilities.shape} does not match existing visibilities {existing_visibilities.shape}."
            )

        # Overwrite the visibilities
        tb_spw.putcol("DATA", new_visibilities)

        # Close the spectral window table
        tb_spw.close()

    # Close the MS
    tb_tool.close()

# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to one if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read in a measurement set and format ready for OSKAR to handle."
    )
    parser.add_argument(
        "--ms",
        help="Measurement set file to process.",
        default="/share/nas2_5/mbowles/data/alma/HTLup_continuum.ms",
    )
    parser.add_argument(
        "-n",
        "--number",
        help="The name of the file under which the formatted data is saved.",
        required=False,
        default=None,
    )
    parser.add_argument("--cont", action="store_true", help="Continue processing")

    ### Parse arguments
    args = parser.parse_args()
    ms_file = Path(args.ms)

    # # Set out index

    ### Add other noise structures here if destired
    # generate_visibilities()
    # add_phase_errors()
    # add_*()

    # print(args.cont)
    # print(f">>>> index {index}")

    ### Write out a new visibility set with the generated
    if N_WORKERS!=1: 
        index = THIS_WORKER
    # else: 
    #     if args.number is not None:
    #         index = int(args.number)
    #     else:
    #         index = 0
    #     if args.cont:  # Continue with next index
    #         index = 0
    #         # Loop through each file in the folder
    #         for file in ms_file.parent.glob("*.fits"):
    #             # Extract the number from the filename
    #             num = int(re.findall(r"\d+", str(file))[-1])
    #             if num:
    #                 # If num is greater than or equal to index, update index
    #                 if num >= index:
    #                     index = num + 1
    #     print(index)
    for i in range(10, 100):
        index = i
        overwrite_visibility(ms_path=str(ms_file))
        image_name = quick_imaging.quick_clean(vis=str(ms_file), index=index)
        quick_imaging.export_fits(image_name=image_name)
