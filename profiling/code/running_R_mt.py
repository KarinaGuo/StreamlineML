import subprocess
import os, csv, glob
import time
import multiprocessing as mp
import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument("base_dir", help="Base directory of processes", type=str,)
args = parser.parse_args()
base_dir = args.base_dir

output_file = os.path.join(base_dir, "log.txt")
in_dir = os.path.join(base_dir, "temp_pred_leaf/subf")
filelist = glob.glob(os.path.join(in_dir, '*.csv'))

def call_dimension_calcs(i):
   start_time = time.time() 
   fileproc = subprocess.check_output(['/home/karina/miniconda3/envs/MLpredictions/bin/Rscript', '--vanilla', "/home/botml/code/dev/profiling/code/leaf_dimension_calculations.R", i], universal_newlines=True, stderr=subprocess.STDOUT)
   end_sheet_time = time.time()
   print(csv_file + ", start_sheet_time:" + str((start_time)) + ", end_sheet_time:" + str((end_sheet_time)) + ", difference:" + str((end_sheet_time - start_time)))
   print(fileproc)
   return fileproc

#results = []
#pool= mp.Pool(processes=20)

for csv_file in filelist:
    print(csv_file)
    #results.append(pool.apply_async(call_dimension_calcs, (csv_file,)))

results = []
pool= mp.Pool(processes=20)

for csv_file in filelist:
    #print(csv_file)
    results.append(pool.apply_async(call_dimension_calcs, (csv_file,)))
    
pool.close()
pool.join()


with open(output_file, 'a') as f_out:
   for r in results:
      print(r.get())
      f_out.write(r.get())
      f_out.write("\n")
   f_out.close()
   
   
#for csv_file in filelist:
#    print(csv_file)
    #results.append(pool.apply_async(call_dimension_calcs, (csv_file,)))    
#    with open(output_file, 'a') as f_out:
#        for r in results:
#            print(r.get())
#            f_out.write(r.get())
#            f_out.write("\n")
#        f_out.close()  
   
#with open(output_file, 'a') as f_out:
  #header = ['filenames','index', 'mask_area_results', 'circle_area_results', 'curvature_results']
#  writer = csv.writer(f_out, delimiter=',')
  #writer.writerow(header)
#  for i in filelist:
#    print("Extracting profile times for", i)
#    fileproc = subprocess.check_output(['/home/karina/miniconda3/envs/MLpredictions/bin/Rscript', '--vanilla', "/home/botml/code/dev/profiling/code/leaf_dimension_calculations.R", i], universal_newlines=True, stderr=subprocess.STDOUT)
#    f_out.write(fileproc)
#    f_out.write("\n")
    #res_list = list(fileproc.split(","))
    #print(fileproc)
#
#f_out.close()
