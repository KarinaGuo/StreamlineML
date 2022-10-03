def predict_leaf(base_dir, loop):
  import sys
  sys.path.append("/home/botml/code/py")
  import model_tools
  import linking_functions
  import os
  
  #Inputs for predict_leaf
  if loop == main:
    img_dir= os.path.join(ase_dir, "temp_image_subset/")
    out_dir= os.path.join(base_dir, "temp_pred/")
  if loop == model:  
    img_dir= os.path.join(base_dir, "input_d2_test/")
    out_dir= os.path.join(base_dir, "d2_pred/")
  training_path = base_dir
  training_name= os.path.join(base_dir, "trim_d2_images/")
  yaml_file = os.path.join(base_dir, "model/d2", "model.yaml")
  weights_file = os.path.join(base_dir, "model/d2", "model_final.pth")
  
  fext = "jpg"
  yaml_zoo=False
  weights_zoo=False
  num_classes=2
  model_tools.visualize_predictions(img_dir, out_dir, fext, training_path, training_name, yaml_file, weights_file, yaml_zoo, weights_zoo, num_classes, score_thresh=0.7, s1=10, s2=14)
  model_tools.predict_in_directory(img_dir, out_dir, fext, yaml_file, weights_file, yaml_zoo, weights_zoo, num_classes, score_thresh=0.7)
  groundtruth_name="test"
  model_predictions_file = os.path.join(out_dir, "_predictions.npy")
  model_tools.predictions_summary(model_predictions_file)

  
  
####
#Creates a mask of the largest component
def CCA():
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
  for i in range(1, numLabels):
    A = stats[i, cv2.CC_STAT_AREA]		
    if keepArea == A:
      componentMask = (labels == i).astype("uint8") * 255
      mask = cv2.bitwise_or(mask, componentMask)
      print("Writing", output_filename_csv_resize)
      pd.DataFrame(mask).to_csv(rs_img_path_csv)
      mask_inv = cv2.bitwise_not(mask)

def extract_leaves(base_dir, loop, CCA):
  import numpy as np
  import os, json, cv2, random
  import torch, torchvision
  import detectron2
  import pandas as pd
  import sys
  sys.path.append("/home/botml/code/py")
  import model_tools
  #
  if loop == "main":
    prediction_file = os.path.join(base_dir, "temp_pred", "_predictions.npy")
    image_directory = os.path.join(base_dir, "temp_image_subset/")
    output_directory = os.path.join(base_dir, "temp_pred_leaf/subf/")
  if loop == "model":
    prediction_file = os.path.join(base_dir, "d2_pred/", "_predictions.npy")
    image_directory = os.path.join(base_dir, "input_d2_test/")
    output_directory = os.path.join(base_dir, "pred_leaf")  
  
  connectivity = 4
  #
  model_predictions = model_tools.open_predictions(prediction_file)
  import glob
  #
  for NSWID in (model_predictions):
    NSWIDsplit = os.path.splitext(NSWID)
    raw_im_fil = glob.glob(os.path.join(image_directory, NSWID))
    #
    for j in enumerate(raw_im_fil):
      raw_im = cv2.imread(j[1])
      #  
      for i in enumerate(model_predictions[NSWID].pred_masks):
        print("Reading", NSWIDsplit[0], "leaf", str(i[0]))
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
          
        if CCA == Y:
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
          for i in range(1, numLabels):
            A = stats[i, cv2.CC_STAT_AREA]		
            if keepArea == A:
              componentMask = (labels == i).astype("uint8") * 255
              mask = cv2.bitwise_or(mask, componentMask)
              print("Writing", output_filename_csv_resize)
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
           
        print ("Writing", output_filename_resize)
        cv2.imwrite(rs_img_path, rs_img)
          
####

def predict_from_classifier (base_dir, loop):
	import matplotlib.pyplot as plt
	import numpy as np
	import pandas as pd
	import torch
	from torch import nn
	from torch import optim
	import torch.nn.functional as F
	import csv
	from csv import writer
	import os
	from torchvision import datasets, transforms, models
 
	if loop == "main":
		train_dir = os.path.join(base_dir, "classifier_training_data/")
		data_dir = os.path.join(base_dir, "temp_pred_leaf/") 
    
	if loop == "model":
		train_dir = os.path.join(base_dir, "input_classifier_data/")
		data_dir = os.path.join(base_dir, "pred_leaf/")
	
	model_file = os.path.join(base_dir, "model/classifier/model.pth")
	out_dir = os.path.join(base_dir, "classifier_results_test.csv")
  
	test_transforms = transforms.Compose([transforms.Resize((500,500)), transforms.ToTensor()])
  
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")	
	model=torch.load(model_file)
	model.eval()
  
	data_train = datasets.ImageFolder(train_dir, transform=test_transforms)
	classes = data_train.classes
  
	def predict_image(image):
		image_tensor = test_transforms(image).float()
		image_tensor = image_tensor.unsqueeze_(0)
		input = image_tensor
		input = input.to(device)
		output = model(input)
		index = output.data.cpu().numpy().argmax()
		return index
  
	if loop == "main":
		def get_images():
			fnames = data.imgs
			indices = list(range(len(data)))
			idx = indices
			loader = torch.utils.data.DataLoader(data, batch_size=len(idx))
			dataiter = iter(loader)
			images = dataiter.next()
			return images, fnames    
		data = datasets.ImageFolder(data_dir, transform=test_transforms)
		to_pil = transforms.ToPILImage()
		images, fnames = get_images()
		colnames = ['filename','pr_class']
		results = pd.DataFrame(columns=colnames)
		indices = list(range(len(data)))
		for ii in range(len(indices)):
			fn  = os.path.basename(fnames[ii][0])
			image = to_pil(images[0][ii])
			pclass = predict_image(image)
			results.loc[len(results)] = [fn, classes[pclass]]
			results.to_csv(out_dir, index=False)
	
	if loop == "model":
		def get_images():
			classes = data.classes
			fnames = data.imgs
			indices = list(range(len(data)))
			idx = indices
			loader = torch.utils.data.DataLoader(data, batch_size=len(idx))
			dataiter = iter(loader)
			images, labels = dataiter.next()
			return images, labels, fnames
		data = datasets.ImageFolder(data_dir, transform=test_transforms)
		to_pil = transforms.ToPILImage()
		images, labels, fnames = get_images()
		colnames = ['filename','pr_class', 'gt_class']
		results = pd.DataFrame(columns=colnames)
		indices = list(range(len(data)))
		for ii in range(len(indices)):
			fn  = os.path.basename(fnames[ii][0])
			image = to_pil(images[ii])
			pclass = predict_image(image)
			lclass = labels[ii].item()
			results.loc[len(results)] = [fn, classes[lclass], classes[pclass]]
			print(fn, classes[lclass], classes[pclass])
			results.to_csv(out_dir, index=False)
      
#####

def removing_files (base_dir):
  import glob, os
  import csv
  import pandas as pd  
  
  directory = os.path.join(base_dir, "temp_pred_leaf/subf")
  class_pr_res = os.path.join(base_dir, "classifier_results_test.csv")
  out_res = os.path.join(base_dir, "temp_filter.csv")
  
  class_res = pd.read_csv(class_pr_res, sep=',')
  filter_res = class_res [class_res[1]=="N"]
  filter_res = filter_res.loc[:,[0]]
  filter_res.to_csv(out_res, index = False)
  
  files_to_delete = set()
  
  with open(out_res, 'r') as m:
    m.readline()
    for i in m:
      if directory in i[0]:
        i[0] = directory +  i[0].split(directory)[-1]
      if i:
        files_to_delete.add(i.replace(',', os.path.sep))

  for i in glob.iglob(os.path.join(directory, "*"), recursive = True):
    for files in files_to_delete:
      print('Removed', files)
      os.remove(os.path.join(directory, files))
      files_to_delete.remove(file_to_delete)

  for files in glob.iglob(os.path.join(directory, "*.jpg"), recursive = True):
    os.remove(files)


#############





##############

def cut_focal_box (label_directory, new_directory, focalbox, classes):
	import os
	import json
	import numpy as np
	import cv2
	import labelme
	import base64
	import glob

	def boundingBox(points):
		min_x = min(point[0] for point in points)
		min_y = min(point[1] for point in points)
		max_x = max(point[0] for point in points)
		max_y = max(point[1] for point in points)
		return min_x, min_y, max_x, max_y

	def get_focalbox_by_shape_name(maskdata, focalbox):
		for i, shape in enumerate(maskdata["shapes"]):
			label = shape["label"]
			if label == focalbox:
				points = shape["points"]
				#points = list(map(int, [points[0][0], points[0][1], points[1][0], points[1][1]]))
				#points = list(map(int, [points[0][1], points[0][0], points[1][1], points[1][0]]))
				#points = list(map(int, [points[0][0], points[0][1], points[1][0], points[1][1]]))
				points = list(map(int, boundingBox(points)))
				print("Found focal box")
				#print(points)
				return points

	def boundingBox(points):
		min_x = min(point[0] for point in points)
		min_y = min(point[1] for point in points)
		max_x = max(point[0] for point in points)
		max_y = max(point[1] for point in points)
		return min_x, min_y, max_x, max_y

	def get_new_shapes(maskdata, focalpoints, use_categories):
		shapes = maskdata["shapes"]
		newshapes = []
		#print(len(shapes))
		if len(shapes) > 0:
			for i in range(len(shapes)):
				#print(i)
				shape = shapes[i]
				label = shape["label"]
				#print(use_categories)
				if label in use_categories :
					newshape = shape
					points = shape["points"]
					#print(len(points))
					if len(points) > 0:
						bounds = boundingBox(points)
						minX = int(bounds[0])
						minY = int(bounds[1])
						maxX = int(bounds[2]) 
						maxY = int(bounds[3]) 
						infocalbox = True
						if minX < focalpoints[0]:
							infocalbox = False
						if maxX > focalpoints[2]:
							infocalbox = False
						if minY < focalpoints[1]:
							infocalbox = False
						if maxY > focalpoints[3]:
							infocalbox = False
						if infocalbox:
							#change contour poitns to fit inside the new bounds
							newpoints = points
							for j in range(len(points)):
								newpoints[j][0] = points[j][0] - focalpoints[0] 
								newpoints[j][1] = points[j][1] - focalpoints[1]
							newshape["points"] = newpoints
							newshapes.append(newshape)
						   # append newshape to newshapes
		return newshapes
	   
	def cut_label_files(fileNameNoExtension, jsonPath, imagePath, cutDir, focalbox, use_categories):
		with open(jsonPath, "r", encoding="utf-8") as read_file:
			maskdata = json.load(read_file)
			if not maskdata["shapes"] or len(maskdata["shapes"]) < 1:
				print("No shapes found")
				return

			# get the focal box (ie, a shape with label name in focalbox)
			focalpoints = get_focalbox_by_shape_name(maskdata, focalbox)
			print(focalpoints)
			# focalpoints: xmin, ymin, xmax, ymax

			# if this exists, crop the image, and start setting up new JSON
			image = cv2.imread(imagePath)
			print("Image shape:")
			print(image.shape)
			#cropImg = image[points[0]:points[2], points[1]:points[3]] 
			cropImg = image[focalpoints[1]:focalpoints[3], focalpoints[0]:focalpoints[2]]
			print("Cropped image shape:")
			print(cropImg.shape)
			i=0 # hack
			newImgName = f"{fileNameNoExtension}_{str(i)}.jpg"
			newJSONName = f"{fileNameNoExtension}_{str(i)}.json"
			newImgPath = os.path.join(cutDir, newImgName)
			newJSONPath = os.path.join(cutDir, newJSONName)
			cv2.imwrite(newImgPath, cropImg) 
				
			imageAsLabelme = labelme.LabelFile.load_image_file(newImgPath)
			imageBase64 = base64.b64encode(imageAsLabelme).decode("utf-8")
			newmask = maskdata
			newmask["imageData"] = imageBase64
			newmask["imagePath"] = newImgName
			newh, neww = cropImg.shape[:2]
			newmask["imageHeight"] = newh
			newmask["imageWidth"] = neww

			newshapes = get_new_shapes(maskdata, focalpoints, use_categories) 
			newmask["shapes"] = newshapes

			with open(newJSONPath, "w", encoding="utf-8") as write_file:
				json.dump(newmask, write_file)
  
	if __name__ == "__main__":
		import os
		import itertools 
		arg_classes = list(itertools.chain(classes))
		labelme_json = glob.glob(os.path.join(label_directory, "*.json"))
		for num, json_file in enumerate(labelme_json):
			fileName = os.path.basename(json_file)
			fileNamesplit = os.path.splitext(fileName)
			fileNameNoExtension = fileNamesplit[0]
			print("Image " + fileNameNoExtension + "...")
			imageFileName = fileNameNoExtension + ".jpg"
			jpg_file = os.path.join(label_directory, imageFileName)
			use_categories=list(itertools.chain(classes))
			cut_label_files(fileNameNoExtension, json_file, jpg_file, new_directory, focalbox, use_categories)
    
####


def lm2coco (labelme_images, output, polyORbb, classes):
	# python lm2coco.py /home/jgb/aiplants/mrs/leaf --output leaf.json --classes 'leaf' --polyORbb 'poly'
	# python lm2coco.py /home/jgb/aiplants/mrs/repstruct --output repstruct.json --classes 'flower' --classes 'fruit' --classes 'bud' --polyORbb 'bb'
	# python /srv/scratch/cornwell/code/py/lm2coco.py /srv/scratch/cornwell/blm/models/detectleaf/data --output data.json --classes 'leaf' --polyORbb 'bb'
#
	import os
	import json
	from labelme import utils
	import numpy as np
	import glob
	import PIL.Image
#
	class labelme2coco(object):
		def __init__(self, labelme_json=[], save_json_path="./coco.json",use_labels=[], polyORbb="poly"):
			"""
			:param labelme_json: the list of all labelme json file paths
			:param save_json_path: the path to save new json
			"""
			self.labelme_json = labelme_json
			self.save_json_path = save_json_path
			self.use_labels = use_labels
			self.polyORbb = polyORbb
			self.images = []
			self.categories = []
			self.annotations = []
			self.label = []
			self.annID = 1
			self.height = 0
			self.width = 0
			self.save_json()
			print(self.use_labels)
		def data_transfer(self):
			for num, json_file in enumerate(self.labelme_json):
				with open(json_file, "r") as fp:
					data = json.load(fp)
					self.images.append(self.image(data, num))
					for shapes in data["shapes"]:
						label = shapes["label"].split("_")
						#print(label)
						#print(self.use_labels)
						#if label[0] in ['partialLeaf']:
						if label[0] in self.use_labels:
							if label not in self.label:
								self.label.append(label)
							points = shapes["points"]
							self.annotations.append(self.annotation(points, label, num, polyORbb))
							self.annID += 1
#
			# Sort all text labels so they are in the same order across data splits.
			self.label.sort()
			for label in self.label:
				self.categories.append(self.category(label))
			for annotation in self.annotations:
				annotation["category_id"] = self.getcatid(annotation["category_id"])
#
		def image(self, data, num):
			image = {}
			img = utils.img_b64_to_arr(data["imageData"])
			height, width = img.shape[:2]
			img = None
			image["height"] = height
			image["width"] = width
			image["id"] = num
			image["file_name"] = data["imagePath"].split("/")[-1]
#
			self.height = height
			self.width = width
#
			return image
#
		def category(self, label):
			category = {}
			category["supercategory"] = label[0]
			category["id"] = len(self.categories)
			category["name"] = label[0]
			return category
#
		def annotation(self, points, label, num, polyORbb):
			annotation = {}
			contour = np.array(points)
			x = contour[:, 0]
			y = contour[:, 1]
			annotation["category_id"] = label[0]  # self.getcatid(label)
			annotation["id"] = self.annID
			annotation["image_id"] = num
			annotation["iscrowd"] = 0
			if polyORbb == "poly":
			   area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
			   annotation["segmentation"] = [list(np.asarray(points).flatten())]
			   annotation["area"] = area
			   annotation["bbox"] = list(map(float, self.getbbox(points)))
			else:
			   bb = list(map(float, self.getbbox(points)))
			   area = (bb[2]) * (bb[3])
			   annotation["segmentation"] = []
			   annotation["area"] = area
			   annotation["bbox"] = bb
#
			return annotation
#
		def getcatid(self, label):
			for category in self.categories:
				if label == category["name"]:
					return category["id"]
			print("label: {} not in categories: {}.".format(label, self.categories))
			exit()
			return -1
#
		def getbbox(self, points):
			polygons = points
			mask = self.polygons_to_mask([self.height, self.width], polygons)
			return self.mask2box(mask)
#
		def mask2box(self, mask):
			index = np.argwhere(mask == 1)
			rows = index[:, 0]
			clos = index[:, 1]
#
			left_top_r = np.min(rows)  # y
			left_top_c = np.min(clos)  # x
#
			right_bottom_r = np.max(rows)
			right_bottom_c = np.max(clos)
#
			return [
				left_top_c,
				left_top_r,
				right_bottom_c - left_top_c,
				right_bottom_r - left_top_r,
			]
#
		def polygons_to_mask(self, img_shape, polygons):
			mask = np.zeros(img_shape, dtype=np.uint8)
			mask = PIL.Image.fromarray(mask)
			xy = list(map(tuple, polygons))
			PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
			mask = np.array(mask, dtype=bool)
			return mask
#
		def data2coco(self):
			data_coco = {}
			data_coco["images"] = self.images
			data_coco["categories"] = self.categories
			data_coco["annotations"] = self.annotations
			return data_coco
#
		def save_json(self):
			print("save coco json")
			self.data_transfer()
			self.data_coco = self.data2coco()
#
			print(self.save_json_path)
			os.makedirs(
				os.path.dirname(os.path.abspath(self.save_json_path)), exist_ok=True
			)
			json.dump(self.data_coco, open(self.save_json_path, "w"), indent=4)
	if __name__ == "__main__":
		import itertools 
		arg_classes = list(itertools.chain(classes))
		polyORbb = polyORbb
		labelme_json = glob.glob(os.path.join(labelme_images, "*.json"))
		print(polyORbb)
		labelme2coco(labelme_json, output, arg_classes, polyORbb)


########

def train_leaf (base_dir):
	import sys
	sys.path.append("/home/botml/code/py")
	import model_tools

	training_path= base_dir
	training_name= os.path.join(base_dir, "trim_d2_data/")
	validation_name= os.path.join(base_dir, "trim_d2_validation/")
	out_dir="model/d2/"
	out_yaml="model.yaml"
	in_yaml="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
	in_weights="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
	in_yaml_zoo=True
	in_weights_zoo=True
	ims_per_batch=20
	base_lr=0.0001
	max_iter=8000
	num_classes=2

	model_tools.train_val_model(training_path, training_name, validation_name, out_dir, out_yaml, in_yaml, in_weights, in_yaml_zoo, in_weights_zoo, ims_per_batch, base_lr, max_iter, num_classes)
 
#########

def train_classifier (base_dir):
	import sys
	sys.path.append("/home/botml/code/py")
	import model_tools
	
	from PIL import ImageFile
	ImageFile.LOAD_TRUNCATED_IMAGES = True
	data_dir   = os.path.join(base_dir, "input_classifier_data")
	val_ratio  = 0.2
	num_epochs = 42 
	model_out  = os.path.join(base_dir, "model/classifier/model.pth")
	model_tools.generic_classifier(data_dir, val_ratio, num_epochs, model_out)

#####

def removing_duplicates():
  import sys
  sys.path.append("/home/botml/code/py")
  import model_tools
  
  model_predictions_file = "/home/karina/test/temp_pred/_predictions.npy"
  duplicates_out_file="/home/jgb/test_duplicates_out.csv"
  model_tools.find_duplicate_predictions(model_predictions_file, duplicates_out_file, 0.7)
 
