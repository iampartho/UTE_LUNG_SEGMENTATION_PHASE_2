import SimpleITK as sitk
import numpy as np
from scipy.ndimage import binary_dilation
import glob
import tqdm
import os

all_directory = glob.glob("/Shared/lss_segerard/parthghosh/data/UTE_new_data_numpy/*/*")

target_filename = "fix_the_mask.py"

for each_dir in tqdm.tqdm(all_directory):

	all_files = os.listdir(each_dir)
	

	if target_filename in all_files:

		filename = each_dir + "/AnatCorrLungs.npy"
		arr = np.load(filename)

		img = arr[:,:,:,0]
		mask = arr[:,:,:,1]

		mask = binary_dilation(mask, structure=np.ones((3,3,3), dtype=bool), iterations=1).astype(np.uint8)

		arr = np.stack((img, mask), axis=-1)

		np.save(filename, arr)

		os.remove(f"{each_dir}/{target_filename}")
	else:
		continue


