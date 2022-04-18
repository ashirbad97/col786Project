import nibabel as nib
import numpy as np 
import matplotlib.pyplot as plt

func_list_filename = 'inputPath.txt'
func_list_file= open(func_list_filename, 'r')

func_list = [line.rstrip("\r\n") for line in func_list_file.readlines()]
num_subjects = len(func_list)

for index,func_file in enumerate(func_list):
	nii_image = nib.load(func_file)
	header = nii_image.header
	data_4D = nii_image.get_fdata()
	num_voxels = data_4D.shape[0] * data_4D.shape[1] * data_4D.shape[2]
	num_volumes = data_4D.shape[3]
	data = data_4D.reshape((num_voxels, num_volumes)).T
	global_signal = np.mean(data, axis = 1).reshape(num_volumes, 1)
	a = np.dot(data.T, global_signal)
	b = np.linalg.norm(global_signal, axis = 0)*np.linalg.norm(data, axis = 0).reshape(num_voxels, 1)
	scale_factors = np.nan_to_num(np.divide(a, b))
	data_wo_global = data - np.dot(global_signal, scale_factors.T)
	means = np.dot(np.ones((num_volumes,1)), np.mean(data_wo_global, axis = 0).reshape(num_voxels, 1).T)
	std = np.dot(np.ones((num_volumes,1)), np.sqrt(np.var(data,axis = 0)).reshape(num_voxels, 1).T)
	normalised_data = np.nan_to_num(np.divide((data_wo_global - means), std))
	volume = normalised_data.T.reshape((data_4D.shape[0], data_4D.shape[1] , data_4D.shape[2], data_4D.shape[3]))
	output_file = nib.Nifti1Image(volume, nii_image.affine, nii_image.header) # generate nifti image
	nib.save(output_file,"TD_Standardized/"+ str(index) +"_standardized.nii.gz")

		
