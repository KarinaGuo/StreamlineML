# python /data/botml/leaf_dimension_classifier/code/extracting_leaves_cropped_iter_v2.py "/data/botml/leaf_dimension_classifier/testing_images_pred/_predictions.npy" "/data/botml/leaf_dimension_classifier/testing_images/" "/data/botml/leaf_dimension_classifier/classifier_training_testdata/"

#python /data/botml/leaf_dimension_classifier/code/extracting_leaves_cropped_iter_v2.py "/home/botml/euc/models/fb2/pred/_predictions.npy" "/home/botml/euc/models/fb2/test/" "/home/karina/test/"

import numpy as np
import os, json, cv2, random
import torch, torchvision
import detectron2
import pandas as pd
import sys
sys.path.append("/home/botml/code/py")
import model_tools
import argparse
import multiprocessing as mp
import time
all_time = time.time()

parser = argparse.ArgumentParser(description="")
parser.add_argument("base_dir", help="", type=str,)
parser.add_argument("loop", help="", type=str,)
parser.add_argument("CCA", help="Output directory of crops.", type=str,)
#parser.add_argument("connectivity", type=int, default=4, help="connectivity for connected component analysis")
args = parser.parse_args()

if args.loop == "main":
	prediction_file = os.path.join(args.base_dir, "temp_pred", "_predictions.npy")
	image_directory = os.path.join(args.base_dir, "temp_image_subset/")
	output_directory = os.path.join(args.base_dir, "temp_pred_leaf/subf/")
if args.loop == "model":
	prediction_file = os.path.join(args.base_dir, "d2_pred/", "_predictions.npy")
	image_directory = os.path.join(args.base_dir, "input_d2_test/")
	output_directory = os.path.join(args.base_dir, "pred_leaf")  

connectivity = 4
#
model_predictions = model_tools.open_predictions(prediction_file)
import glob
#

def extract_leaves_from_sheet(NSWID, model_predictions, image_directory, output_directory, CCA, connectivity):
	#for NSWID in (model_predictions):
	import time
	start_time = time.time()
	import numpy as np
	import os, json, cv2, random
	import torch, torchvision
	import detectron2
	import pandas as pd
	import sys
	sys.path.append("/home/botml/code/py")
	import model_tools
	load_time = time.time()
  
	for NSWID in (model_predictions):
		load_time = time.time()
		NSWIDsplit = os.path.splitext(NSWID)
		raw_im_fil = glob.glob(os.path.join(image_directory, NSWID))

		for j in enumerate(raw_im_fil):
			raw_im = cv2.imread(j[1])
			  #  
			for i in enumerate(model_predictions[NSWID].pred_masks):
				#print("Reading", NSWIDsplit[0], "leaf", str(i[0]))
				output_filename_resize = f"{NSWIDsplit[0]}_{str(i[0])}.jpg"
				output_filename_csv_resize = f"{NSWIDsplit[0]}_{str(i[0])}.csv"
				output_resize_dir = output_directory
				rs_img_path = os.path.join(output_resize_dir, output_filename_resize)
				rs_img_path_csv = os.path.join(output_resize_dir, output_filename_csv_resize)
					
				array = i[1]
				in_leaf = cv2.imread(j[1])
				in_leaf[~array,:] = [0,0,0]
					
				rows = np.any(in_leaf, axis=1)
				cols = np.any(in_leaf, axis=0)
				rmin, rmax = np.where(rows)[0][[0, -1]]
				cmin, cmax = np.where(cols)[0][[0, -1]]

				crop_rmin = int(rmin*0.9)
				crop_rmax = int(rmax*1.1)
				crop_cmin = int(cmin*0.9)
				crop_cmax = int(cmax*1.1)

				rsize = crop_rmax - crop_rmin
				csize = crop_cmax - crop_cmin
					
				if rsize >= csize:
					rpad = rsize // 10
					cpad = rpad + (rsize - csize) // 2
				if csize > rsize:
					cpad = csize // 10
					rpad = cpad + (csize - rsize) // 2
				  
				if CCA == "Y":
					gray = cv2.cvtColor(in_leaf, cv2.COLOR_BGR2GRAY)
					thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
					output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
					(numLabels, labels, stats, centroid) = output
					mask = np.zeros(array.shape, dtype="uint8")
					area = []
					  
				for i in range(1, numLabels):
					A = stats[i, cv2.CC_STAT_AREA]
					area.append(A)
					keepArea = max(area)	
					if keepArea == A:
						componentMask = (labels == i).astype("uint8") * 255
						mask = cv2.bitwise_or(mask, componentMask)
						#print("Writing", output_filename_csv_resize)
						pd.DataFrame(mask).to_csv(rs_img_path_csv)
						mask_inv = cv2.bitwise_not(mask) 
				
				res = cv2.bitwise_and(raw_im, raw_im, mask=mask)
				gray_bg = cv2.cvtColor(raw_im, cv2.COLOR_BGR2GRAY)
				background = cv2.bitwise_and(gray_bg, gray_bg, mask = mask_inv)
				background = np.stack((background,)*3, axis=-1)
				img_ca = cv2.add(res, background)
				  
				crop_img = img_ca[crop_rmin:crop_rmax, crop_cmin:crop_cmax]
				color = [0, 0, 0]
				pad_img = cv2.copyMakeBorder(crop_img,rpad,rpad,cpad,cpad,cv2.BORDER_CONSTANT,value=color)     
				rs_img = cv2.resize(pad_img, (500, 500))
					   
				#print ("Writing", output_filename_resize)
				cv2.imwrite(rs_img_path, rs_img)
      
		end_sheet_time = time.time()
		print(NSWID + ", start_sheet_time:" + str((start_time)) + ", end_sheet_time:" + str((end_sheet_time)) + ", difference:" + str((end_sheet_time - start_time)))

#
#import multiprocessing as mp

# Step 1: Init multiprocessing.Pool()
#pool = mp.Pool(mp.cpu_count())

# Step 2: `pool.apply` the `howmany_within_range()`
#results = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in data]
# results = [pool.apply(extract_leaves_from_sheet, args=(NSWID, model_predictions, image_directory, output_directory, CCA, connectivity)) for NSWID in (model_predictions)]

# Step 3: Don't forget to close
#pool.close()

#pool = mp.Pool(processes=10)
#results = [pool.apply(extract_leaves_from_sheet, args=(NSWID, model_predictions, image_directory, output_directory, args.CCA, connectivity)) for NSWID in model_predictions]
#pool.close()

pool = mp.Pool(processes=5)
for NSWID in (model_predictions): 
   pool.apply_async(extract_leaves_from_sheet, (NSWID,), dict(model_predictions=model_predictions, image_directory=image_directory, output_directory=output_directory, CCA=args.CCA, connectivity=connectivity)) 
pool.close()
pool.join()

print(str((all_time)) + " " + str((time.time() - all_time)))
