import os
import numpy as np
import imageio
## create a folder in the given path
def create_dir_if_not_exist(d):
    if not os.path.exists(d):
        os.mkdir(d)

## create a local folder to store all the images
def create_dir_if_not_exist(d):
    if not os.path.exists(d):
        os.mkdir(d)         
        
## process the raw files with the following procedure
## reshape the files and convert them into png format for modeling
def process_mris(files, target_dir):
    for f in files:
        print(f)
        mris = np.fromfile(open(f, 'rb'), dtype='>u2')\
            .reshape((176, 208, 176))[:, :, np.arange(1, 176, 2)].transpose((2, 0, 1))
        for i, mri in enumerate(mris):
            new_fname = "_".join(os.path.basename(f).split('.')[0].split('_')[:8])+"_%i.png" % i
            if np.max(mri) <= 255:
                imageio.imsave(os.path.join(target_dir, new_fname), mri.astype(np.uint8))
            else:
                imageio.imsave(os.path.join(target_dir, new_fname), mri)
    return

## process mask files and label the files into 4 classes
def bin_mask(raw_segmentation):
    raw_segmentation[raw_segmentation <= 150] = 0
    raw_segmentation[np.where((150 < raw_segmentation) & (raw_segmentation <= 400))] = 1
    raw_segmentation[np.where((400 < raw_segmentation) & (raw_segmentation <= 625))] = 2
    raw_segmentation[raw_segmentation > 625] = 3
    return raw_segmentation


def process_labels(files, target_dir):
    for f in files:
        tmp = np.fromfile(open(f, 'rb'), dtype='>u2').reshape(
            (176, 208, 88)).transpose((2, 0, 1))
        masks = bin_mask(tmp)
        for i, mask in enumerate(masks):
            new_fname = "_".join(os.path.basename(f).split('.')[0].split('_')[:8])+"_%i_mask.png" % i
            imageio.imsave(os.path.join(target_dir, new_fname), mask.astype(np.uint8))
    return