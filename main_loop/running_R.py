import subprocess
import os, csv, glob

base_dir = os.getcwd()

output_file = os.path.join(base_dir, "final_results.csv")
in_dir = os.path.join(base_dir, "temp_pred_leaf/")

filelist = glob.glob(os.path.join(in_dir, '*.csv'))

with open(output_file, 'a') as f_out:
  header = ['filenames', 'index', 'circle_area_results', 'curvature_results', 'mask_area_results']
  writer = csv.writer(f_out, delimiter=',')
  writer.writerow(header)
  for i in filelist:
    fileproc = subprocess.check_output(['/usr/bin/Rscript', '--vanilla', "/home/botml/code/dev/main_loop/temp.R", i], universal_newlines=True, stderr=subprocess.STDOUT)
    res_list = list(fileproc.split(","))
    writer.writerow(res_list)