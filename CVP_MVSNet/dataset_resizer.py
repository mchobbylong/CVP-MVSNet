from PIL import Image
import numpy as np
import os
import shutil
import sys
import multiprocessing as mp
import argparse
import cv2


def read_pfm(filename):
	file = open(filename, 'rb')
	color = None
	width = None
	height = None
	scale = None
	endian = None
	
	header = file.readline().decode('utf-8').rstrip()
	if header == 'PF':
		color = True
	elif header == 'Pf':
		color = False
	else:
		raise Exception('Not a PFM file.')

	dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
	if dim_match:
		width, height = map(int, dim_match.groups())
	else:
		raise Exception('Malformed PFM header.')

	scale = float(file.readline().rstrip())
	if scale < 0:  # little-endian
		endian = '<'
		scale = -scale
	else:
		endian = '>'  # big-endian

	data = np.fromfile(file, endian + 'f')
	shape = (height, width, 3) if color else (height, width)

	data = np.reshape(data, shape)
	data = np.flipud(data)
	file.close()
	return data, scale


def save_pfm(filename, image, scale=1):
	file = open(filename, "wb")
	color = None

	image = np.flipud(image)

	if image.dtype.name != 'float32':
		raise Exception('Image dtype must be float32.')

	if len(image.shape) == 3 and image.shape[2] == 3:  # color image
		color = True
	elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
		color = False
	else:
		raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

	file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
	file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

	endian = image.dtype.byteorder

	if endian == '<' or endian == '=' and sys.byteorder == 'little':
		scale = -scale

	file.write(('%f\n' % scale).encode('utf-8'))

	image.tofile(file)
	file.close()



def resize_image(tup):
	old_img, new_img, scales = tup
	im = Image.open(old_img)
	im = im.resize((int(im.width * scales[0]), int(im.height * scales[1])), resample=Image.LANCZOS)
	im.save(new_img)


def resize_depth_map(tup):
	old_dep, new_dep, scales = tup
	pfm = read_pfm(old_dep)[0]
	pfm = cv2.resize(pfm, None, fx=scales[0], fy=scales[1], interpolation=cv2.INTER_LANCZOS4)
	save_pfm(new_dep, pfm)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Resize DTU dataset.')
	parser.add_argument('dataset_path', type=str, help='')
	parser.add_argument('-o', '--output_path', type=str, default=None, help='Output dataset path')
	parser.add_argument('--resize_depth', default=False, action='store_true',
		help='Resize depth maps')
	parser.add_argument('--wh_scales', type=float, nargs='+', help='Resize scale factors along width and height direction')
	args = parser.parse_args()

	scales = args.wh_scales
	if len(scales) < 2: scales += scales

	dataset_path = args.dataset_path
	cam_dir = os.path.join(dataset_path, 'Cameras')
	img_dir = os.path.join(dataset_path, 'Rectified')
	dep_dir = os.path.join(dataset_path, 'Depths')

	new_dataset_path = args.output_path or dataset_path.rstrip('\\', '/') + '_resized'
	new_cam_dir = os.path.join(new_dataset_path, 'Cameras')
	new_img_dir = os.path.join(new_dataset_path, 'Rectified')
	new_dep_dir = os.path.join(new_dataset_path, 'Depths')

	# Resize camera profiles
	os.makedirs(new_cam_dir, exist_ok=True)
	for camfile in os.listdir(cam_dir):
		if camfile.find('cam') == -1: continue
		with open(os.path.join(cam_dir, camfile), 'r') as f:
			lines = [line.strip() for line in f.readlines()]
		intrinsic = np.fromstring(' '.join(lines[7:10]), dtype=float, sep=' ').reshape((3, 3))
		intrinsic_scale = np.array([
			[scales[0], 1, scales[0]],
			[1, scales[1], scales[1]],
			[1, 1, 1],
		])
		intrinsic = (intrinsic * intrinsic_scale).tolist()

		lines = [line + '\n' for line in lines]
		with open(os.path.join(new_cam_dir, camfile), 'w') as f:
			f.writelines(lines[0:7])
			for l in intrinsic:
				f.write('%.6f %.6f %.6f\n' % tuple(l))
			f.writelines(lines[10:])

	# Read scan list
	try:
		with open(os.path.join(dataset_path, 'scan_list_test.txt'), 'r') as f:
			scans = [line.strip() for line in f.readlines()]
	except:
		scans = ['scan%d' % i for i in set(range(1,129)).difference(range(78,82))]

	# Resize images
	queue = []
	for scan in scans:
		cur_img_dir = os.path.join(img_dir, scan)
		cur_new_img_dir = os.path.join(new_img_dir, scan)

		os.makedirs(cur_new_img_dir, exist_ok=True)
		for imgfile in os.listdir(cur_img_dir):
			queue.append((os.path.join(cur_img_dir, imgfile), os.path.join(cur_new_img_dir, imgfile), scales))
	p = mp.Pool(processes=mp.cpu_count())
	p.map(resize_image, queue)

	# Resize depth maps
	if args.resize_depth:
		queue = []
		for scan in scans:
			cur_dep_dir = os.path.join(dep_dir, scan)
			cur_new_dep_dir = os.path.join(new_dep_dir, scan)

			os.makedirs(cur_new_dep_dir, exist_ok=True)
			for depfile in os.listdir(cur_dep_dir):
				if os.path.splitext(depfile)[1].lower() != '.pfm': continue
				queue.append((os.path.join(cur_dep_dir, depfile), os.path.join(cur_new_dep_dir, depfile), scales))
		p.map(resize_depth_map, queue)

	# Copy other files
	shutil.copyfile(os.path.join(cam_dir, 'pair.txt'), os.path.join(new_cam_dir, 'pair.txt'))
	shutil.copyfile(os.path.join(dataset_path, 'scan_list_test.txt'), os.path.join(new_dataset_path, 'scan_list_test.txt'))
