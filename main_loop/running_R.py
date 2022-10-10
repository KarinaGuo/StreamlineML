import subprocess
import os, csv, glob

import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument("base_dir", help="Base directory of processes", type=str,)
args = parser.parse_args()
base_dir = args.base_dir

output_file = os.path.join(base_dir, "final_results.csv")
in_dir = os.path.join(base_dir, "temp_pred_leaf/subf")
print(in_dir)
filelist = glob.glob(os.path.join(in_dir, '*.csv'))

with open(output_file, 'a') as f_out:
  #header = ['filenames','index', 'mask_area_results', 'circle_area_results', 'curvature_results']
  writer = csv.writer(f_out, delimiter=',')
  #writer.writerow(header)
  for i in filelist:
    print("Extracting trait values for", i)
    fileproc = subprocess.check_output(['/home/karina/miniconda3/envs/MLpredictions/bin/Rscript', '--vanilla', "/home/botml/code/dev/main_loop/leaf_dimension_calculations.R", i], universal_newlines=True, stderr=subprocess.STDOUT)
    res_list = list(fileproc.split(","))
    writer.writerow(res_list)


f_out.close()
