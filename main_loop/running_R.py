import argparse
import subprocess
import os, csv, glob

parser = argparse.ArgumentParser(description="")
parser.add_argument("base_dir", help="Base directory of processes", type=str,)
#parser.add_argument("out", help="CSV output file to append to", type=str,)
args = parser.parse_args()

output_file = os.path.join(base_dir, "final_results.csv")
in_dir = os.path.join(base_dir, "temp_pred_leaf/")

filelist = glob.glob(os.path.join(in_dir, '*.csv')

for i in filelist:
  with open(output_file, 'a') as f_out:
    writer = csv.writer(f_out)
    fileproc = subprocess.check_output(['/usr/bin/Rscript', '--vanilla', "/home/botml/code/dev/main_loop/temp.R", i], universal_newlines=True, stderr=subprocess.STDOUT)
    writer.writerow([fileproc])
    print(fileproc)



path_input = '/home/c.blanes/pgp/training.txt'
obabel = '/usr/bin/obabel '
  
with open(path_input, 'r') as f_in, open(path_input, 'a') as f_out:
 
    for line in f_in.readlines():
 
        smiles = line.split(',')[2].strip('\n')
        print(smiles)
         
        launch = obabel + '-:\'%s\' -otxt --append TPSA'%(smiles)
         
        subprocess.call(launch, shell=True)
         
        f_out.write(f'{launch}\n')

with subprocess.check_output(['/usr/bin/Rscript', '--vanilla', "/home/botml/code/dev/main_loop/temp.R", base_dir], universal_newlines=True, stdout=subprocess.PIPE) as fileproc:
  table = proc.stdout.read()
