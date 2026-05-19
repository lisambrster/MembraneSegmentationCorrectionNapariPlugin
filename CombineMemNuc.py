
#  combine membrane and nuclei raw channels - these are ADDED - not two channel - for use with Napari
#  use this to correct segmentation

#  conda activate micro-sam

import numpy as np
import tifffile as tiff
import os
import yaml
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_name', type=str,  default='.',
                        help='configuration name')

    args = parser.parse_args()
    config_name = args.config_name

    # read in config file
    config_path = '../'
    print('Config ', config_name)
    with open(os.path.join(config_path, config_name, 'config.yaml'), 'r') as file:
        config_opts = yaml.safe_load(file)


    start_frame = config_opts['register_begin_frame']
    end_frame = config_opts['register_end_frame']

    nframes = end_frame - start_frame + 1
    print('start frame:', start_frame)
    print('end frame:', end_frame)

    out_path = config_opts["output_path"]
    if (not os.path.exists(out_path)):
        os.makedirs(out_path)
    out_path = out_path + '/Raw_data/' + 'MemNucCombo'
    if (not os.path.exists(out_path)):
        os.makedirs(out_path)


    nuc_path = config_opts["nuc_path"]
    mem_path = config_opts["membrane_path"]

    for iframe in range(start_frame, end_frame):
        mem_name = mem_path % iframe
        nuc_name = nuc_path % iframe
        mem_img = tiff.imread(mem_name)
        nuc_img = tiff.imread(nuc_name)

        mem_img = mem_img + nuc_img

        out_name = "MemNuc_%05d.tif" % iframe
        out_name = os.path.join(out_path, out_name)
        tiff.imwrite(out_name, mem_img)
        print('writing ' + out_name)