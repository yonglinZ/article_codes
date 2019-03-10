import sys
import time
import getopt
import os
import math
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from PIL import Image as PILImage

# Path of the Caffe installation.
_CAFFE_ROOT = "../"

# Model definition and model file paths
_MODEL_DEF_FILE = "model/pspnet_ADE20K.prototxt"  # Contains the network definition
_MODEL_FILE = "model/pspnet_ADE20K.caffemodel"  # Contains the trained weights.

sys.path.insert(0, _CAFFE_ROOT + "python")
import caffe

_MAX_DIM = 473

CITY = 'SH'


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.

    Args:
        num_cls: Number of classes

    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in xrange(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def crfrnn_segmenter(model_def_file, model_file, gpu_device, inputs):
    """ Returns the segmentation of the given image.

    Args:
        model_def_file: File path of the Caffe model definition prototxt file
        model_file: File path of the trained model file (contains trained weights)
        gpu_device: ID of the GPU device. If using the CPU, set this to -1
        inputs: List of images to be segmented 

    Returns:
        The segmented image
    """

    assert os.path.isfile(model_def_file), "File {} is missing".format(model_def_file)
    assert os.path.isfile(model_file), ("File {} is missing. Please download it using "
                                        "./download_trained_model.sh").format(model_file)

    if gpu_device >= 0:
        caffe.set_device(gpu_device)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(model_def_file, model_file, caffe.TEST)

    num_images = len(inputs)
    num_channels = inputs[0].shape[2]
    assert num_channels == 3, "Unexpected channel count. A 3-channel RGB image is exptected."

    caffe_in = np.zeros((num_images, num_channels, _MAX_DIM, _MAX_DIM), dtype=np.float32)
    for ix, in_ in enumerate(inputs):
        caffe_in[ix] = in_.transpose((2, 0, 1))

    start_time = time.time()
    out = net.forward_all(**{net.inputs[0]: caffe_in})
    end_time = time.time()

    print("Time taken to run the network: {:.4f} seconds".format(end_time - start_time))
    predictions = out[net.outputs[0]]

    return predictions[0].argmax(axis=0).astype(np.uint8)


def run_crfrnn(input_file, output_file, gpu_device):
    """ Runs the CRF-RNN segmentation on the given RGB image and saves the segmentation mask.

    Args:
        input_file: Input RGB image file (e.g. in JPEG format)
        output_file: Path to save the resulting segmentation in PNG format
        gpu_device: ID of the GPU device. If using the CPU, set this to -1
    """

    input_image = 255 * caffe.io.load_image(input_file)
    input_image = resize_image(input_image)

    image = PILImage.fromarray(np.uint8(input_image))
    image = np.array(image)

    palette = get_palette(256) # 256 origion
    #PIL reads image in the form of RGB, while cv2 reads image in the form of BGR, mean_vec = [R,G,B] 
    mean_vec = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    mean_vec = mean_vec.reshape(1, 1, 3)

    # Rearrange channels to form BGR
    im = image[:, :, ::-1]
    # Subtract mean
    im = im - mean_vec

    # Pad as necessary
    cur_h, cur_w, cur_c = im.shape
    pad_h = _MAX_DIM - cur_h
    pad_w = _MAX_DIM - cur_w
    im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    # Get predictions
    # very important
    segmentation = crfrnn_segmenter(_MODEL_DEF_FILE, _MODEL_FILE, gpu_device, [im])
    segmentation = segmentation[0:cur_h, 0:cur_w]

    output_im = PILImage.fromarray(segmentation)
    output_im.putpalette(palette)
    # attention!!!!!!!!~~~~~~~~~~~~~~*********&*&&&&&&&  375
    output_im = output_im.resize((1000,375), resample=PILImage.BILINEAR)
    output_im.save(output_file)

    return segmentation # return segmentaion mask


def resize_image(image):
    """ Resizes the image so that the largest dimension is not larger than 500 pixels.
        If the image's largest dimension is already less than 500, no changes are made.

    Args:
        Input image

    Returns:
        Resized image where the largest dimension is less than 500 pixels
    """

    width, height = image.shape[0], image.shape[1]
    max_dim = max(width, height)

    if max_dim > _MAX_DIM:
        if height > width:
            ratio = float(_MAX_DIM) / height
        else:
            ratio = float(_MAX_DIM) / width
        image = PILImage.fromarray(np.uint8(image))
        image = image.resize((int(height * ratio), int(width * ratio)), resample=PILImage.BILINEAR)
        image = np.array(image)

    return image


#def main(argv):
def main(inputs,outputs,gpu_device):
    """ Main entry point to the program. """

    #input_file = "/home/lzhpc/PSPNET-cudnn5/demo/images/5.png"
    #output_file = "/home/lzhpc/PSPNET-cudnn5/demo/images/5_rs.png"



    #gpu_device = 1  # Use -1 to run only on the CPU, use 0-3[7] to run on the GPU

    # try:
    #     opts, args = getopt.getopt(argv, 'hi:o:g:', ["ifile=", "ofile=", "gpu="])
    # except getopt.GetoptError:
    #     print("crfasrnn_demo.py -i <input_file> -o <output_file> -g <gpu_device>")
    #     sys.exit(2)

    # for opt, arg in opts:
    #     if opt == '-h':
    #         print("crfasrnn_demo.py -i <inputfile> -o <outputfile> -g <gpu_device>")
    #         sys.exit()
    #     elif opt in ("-i", "ifile"):
    #         input_file = arg
    #     elif opt in ("-o", "ofile"):
    #         output_file = arg
    #     elif opt in ("-g", "gpudevice"):
    #         gpu_device = int(arg)

    #print("Input file: {}".format(input_file))
    #print("Output file: {}".format(output_file))
    #if gpu_device >= 0:
    #    print("GPU device ID: {}".format(gpu_device))
    #else:
    #    print("Using the CPU (set parameters appropriately to use the GPU)")
    re = run_crfrnn(inputs, outputs, gpu_device)
    scene_parsed = calculator(re)
    return scene_parsed

# calculate GVI, SVF, BVF and so on.
def calculator(input):
	siren = Series(np.zeros(150),index=range(0,150)) # (0,149)150 dimension to 150 objects
	fla = input.flatten()
	tot = len(fla) # total pixel number
	lifla = list(fla)
	redupli = set(lifla) # reduplicate
	#print redupli
	
	for i,redu in enumerate(redupli):
		co = lifla.count(redu) # count pixel number of 'redu'(e.g tree) object
		siren[int(redu)] = float(co)/float(tot)	 # ratio calculation
		#print int(redu),float(co)/float(tot)
	
	siren.index = range(1,151)
	#print siren
	return siren	

if __name__ == "__main__":
    #main(sys.argv[1:])
    time_start = time.time()
    scene_parsed_li = []
    TI_li = []
    root = os.getcwd()
    folder = CITY+'_images' # if have new tasks, please alter 'images'
    path = os.path.join(root,folder)
    output_path = os.path.join(root,folder+'_PSPnet_res')# /home/lzhpc/PSPNET-cudnn5/demo/images/
    if os.path.exists(output_path): # create PSPnet mask outputs' folder
    	pass
    else:
    	os.makedirs(output_path)

    if os.listdir(path) != []:
    	image_li = range(len(os.listdir(path)))
    	image_out_li = range(len(os.listdir(path)))
    	for i, pa in enumerate(os.listdir(path)):
    		im_path = os.path.join(path, pa)
    		out_path = os.path.join(output_path,'res_'+pa)
    		image_li[i] = im_path
    		image_out_li[i] = out_path
    		#print im_path, out_path
    else:
    	print 'No images exist in the folder!' 

    if image_li != [] and image_out_li != []:
    	for i, im in enumerate(image_li):
    		try:
    			time_s = time.time()
    			scene_parsed = main(im, image_out_li[i], 0)# GPU_device 0
    			time_e = time.time()
    			ti = time_e - time_s# time record of each loop
    			scene_parsed_li.append(list(scene_parsed))# record 150 dimension ratio results
    			TI_li.append(ti)
    		except Exception, e:
    			print e
    else:
    	print 'No image files in path:' +path+' exists'
    #main(inputs)
    
    series_image = Series(image_li)
    series_TI = Series(TI_li)
    series_scene = scene_parsed_li
    #print series_image,series_TI,series_scene
    df_out =  DataFrame([])
    df_out['impath'] = series_image
    df_out['time'] = series_TI
    df_out['res'] = series_scene
    #print df_out
    # df_out = DataFrame([series_image,series_TI,scene_parsed_li]) this way went wrong
    # and i don't know why.
  
    result_file_name = CITY+'_pspnet_seg_res.csv'
    df_out.to_csv(os.path.join('results',result_file_name))
    time_end = time.time()
    print(time_end - time_start)

    
