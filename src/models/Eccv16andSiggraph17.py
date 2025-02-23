import matplotlib.pyplot as plt
import torch

from Eccv16andSiggraph17_utils import *
from Eccv16andSiggraph17_utils.util import load_img, preprocess_img, postprocess_tens

from os import listdir
from os.path import isfile, join

colorizer_eccv16 = colorizers.eccv16().eval()
#colorizer_siggraph17 = colorizers.siggraph17().eval()


# DIRECTORY INFORMATION
bnw_input_dir = '../../img/original/test/'
bnw_output_dir = '../../img/colorized/siggraph/test'

onlyfiles = [f for f in listdir(bnw_input_dir) if (isfile(join(bnw_input_dir, f)) and f != '.DS_Store')]

count = 0
for i in onlyfiles:
    count += 1
    if count % 100 == 0:
        print(count)
    img = load_img(join(bnw_input_dir, i))
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
    # colorizer outputs 256x256 ab map
    # resize and concatenate to original L channel
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig, 0*tens_l_orig), dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
    out_path = join(bnw_output_dir, i)
    plt.imsave(out_path, out_img_eccv16)

