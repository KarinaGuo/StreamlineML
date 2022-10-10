import subprocess
import os, csv, glob

import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument("base_dir", help="Base directory of processes", type=str,)
args = parser.parse_args()
base_dir = args.base_dir

output_file = os.path.join(base_dir, "final_results.csv")

with open(output_file, 'w') as f_out:
  header = ['filenames','index', 'mask_area_results', 'circle_area_results', 'curvature_results']
  writer = csv.writer(f_out, delimiter=',')
  writer.writerow(header)


f_out.close()
