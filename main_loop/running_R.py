import subprocess
import os, csv, glob

# this was re-inserted mostly to make it easier for me to run
import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument("base_dir", help="Base directory of processes", type=str,)
args = parser.parse_args()
base_dir = args.base_dir
#base_dir = os.getcwd()

output_file = os.path.join(base_dir, "final_results.csv")
in_dir = os.path.join(base_dir, "temp_pred_leaf/")
print(in_dir)
filelist = glob.glob(os.path.join(in_dir, '*.csv'))

with open(output_file, 'a') as f_out:
  header = ['filenames', 'index', 'circle_area_results', 'curvature_results', 'mask_area_results']
  writer = csv.writer(f_out, delimiter=',')
  writer.writerow(header)
  for i in filelist:
    print(i)
    fileproc = subprocess.check_output(['/usr/bin/Rscript', '--vanilla', "/home/jgb/tmp/kg/StreamlineML/main_loop/leaf_dimension_calculations.R", i], universal_newlines=True, stderr=subprocess.STDOUT)
    res_list = list(fileproc.split(","))
    writer.writerow(res_list)
