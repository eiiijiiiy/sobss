import seg
from tqdm import tqdm
import numpy as np
import argparse

# root_dir = "/media/y/18T/parallelism_2022/revision_exper"
root_dir = "/Volumes/SSD/yijie_revision_exper"

parser = argparse.ArgumentParser()
parser.add_argument('--t', type=str, required=True)
args = parser.parse_args()

assert args.t in 'mi'
if args.t == 'm':
    root_dir = "{}/3-select_primary_normal_manhattan_revision".format(root_dir)
else:
    root_dir = "{}/4-select_primary_normal_incline_roof_revision".format(root_dir)

# params = [0.5, 1, 2, 4]
params = [2]

for p in tqdm(params):
    options = {'ps': p, 'dxi': p, 'xi': p, 'zci': p, 'nmi': np.pi/9}

    options['pcd_dir'] = "{}/3-voted".format(root_dir)
    # options['out_dir'] = "{}/sensitivity/coarse_segmentation/mesh/{:.1f}".format(root_dir, p)
    # options['param_out_dir'] = "{}/sensitivity/coarse_segmentation/param/{:.1f}".format(root_dir, p)
    # options['log_path'] = "{}/sensitivity/coarse_segmentation/log/{:.1f}.txt".format(root_dir, p)

    options['out_dir'] = "{}/sensitivity/coarse_segmentation_tmp/mesh/{:.1f}".format(root_dir, p)
    options['group_dir'] = "{}/sensitivity/coarse_segmentation_tmp/group/{:.1f}".format(root_dir, p)
    options['param_out_dir'] = "{}/sensitivity/coarse_segmentation_tmp/param/{:.1f}".format(root_dir, p)
    options['log_path'] = "{}/sensitivity/coarse_segmentation_tmp/log/{:.1f}.txt".format(root_dir, p)

    seg.coarse_segment(options)
