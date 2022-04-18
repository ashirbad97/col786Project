from distutils import filelist
import numpy as np
import nibabel
from nilearn import datasets
# from nilearn.maskers import NiftiMapsMasker
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure, group_sparse_covariance
import scipy.io
from nilearn import plotting
import matplotlib.pyplot as plt
import os
import networkx as nx
from networkx.algorithms import community



def generateKernel(kernel_size, fwhm, pix_dim):
    ind_var = np.arange(-kernel_size, kernel_size + 1)
    sigma = fwhm / np.sqrt(8 * np.log(2)) / (float(pix_dim)) # calculating standard deviation using fwhm 
    kernel = np.exp(-(ind_var**2) / (2 * (sigma**2))) # generating values for gaussian kernel
    return kernel/sum(kernel) # returning normalised kernel 

# Dead Function
def getSamplePixelDim(sampleFile):
    pixelDim = nibabel.load(sampleFile).header["pixdim"]
    return pixelDim

def convolveWithKernel(kernel,windowSize):
    one = np.ones(windowSize)
    # also known as tapred window 
    convolvedThing = np.convolve(kernel,one)
    # print(convolvedThing.shape)
    return convolvedThing

def getParcellations(sampleFile):
    # Retrieve the atlas and the data
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    # Loading atlas image stored in 'maps'
    atlas_filename = atlas.maps #atlas['maps']
    # Loading atlas data stored in 'labels'
    labels = atlas.labels #atlas['labels']
    print(len(labels))
    # Load the functional datasets
    # data = datasets.fetch_development_fmri(n_subjects=1)
    data = sampleFile
    # print('First subject resting-state nifti image (4D) is located at: %s' %
        # data)
    # Extract the time series
    # masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True,
    #                         memory='nilearn_cache', verbose=5)
    masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
                            memory='nilearn_cache', verbose=0)
    
    # masker.fit(data)
    time_series = masker.fit_transform(data)
    return [time_series,atlas,labels]

def copyConvolvedThing(convolvedThing,noParcellations):
    one = np.ones((noParcellations,1))
    copyConvolvedthing = np.dot(one,convolvedThing)
    return copyConvolvedthing

def getWindows(timeseriesMatrix,copyThing,windowSize,sliceSizeStep):
    timeseriesMatrixT = timeseriesMatrix.T
    # print(timeseriesMatrixT.shape)
    windowEnd = 0
    # windowsStart will increment by the value of teh sliceSizeStep
    windowedView = np.ndarray((timeseriesMatrixT.shape[0],windowSize, (timeseriesMatrixT.shape[1]-windowSize+sliceSizeStep)//sliceSizeStep))
    window_num = 0
    for windowStart in range(0,timeseriesMatrixT.shape[1],sliceSizeStep):
        windowEnd = windowStart + windowSize
        if windowEnd < timeseriesMatrixT.shape[1]:
            # print(windowStart, windowEnd)
            sliceThing = np.multiply(timeseriesMatrixT[:,windowStart:windowEnd], copyThing)
            # np.append(windowedView, sliceThing)
            windowedView[:,:,window_num] = sliceThing
            window_num = window_num + 1
    # print(windowedView.shape)
    return windowedView

def measuringCorrelation(windowThings):
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlationMatrices = np.ndarray((windowThings.shape[0],windowThings.shape[0],windowThings.shape[2]))
    for windowNum in range(windowThings.shape[2]):
        time_series = windowThings[:,:,windowNum].T
        correlationMatrices[:,:,windowNum] = correlation_measure.fit_transform([time_series])[0]
    return correlationMatrices

def get_HO_coords(atlas):
    atlas_img = atlas['maps']
    # all ROIs except the background
    values = np.unique(atlas_img.get_data())[1:] 
    # iterate over Harvard-Oxford ROIs
    coords = []
    for v in values:
        data = np.zeros_like(atlas_img.get_data())
        data[atlas_img.get_data() == v] = 1
        xyz = plotting.find_xyz_cut_coords(nibabel.Nifti1Image(data, atlas_img.affine))
        coords.append(xyz)
    return coords

def plotCorrelationMatrix(correlation_matrix,atlas,labels):
    np.fill_diagonal(correlation_matrix, 0)
    plotting.plot_matrix(correlation_matrix,  labels = labels, colorbar=True,
                     vmax=0.9, vmin=0.3)
    # coords = atlas.region_coords
    # only for harvard oxford atlas 
    coords = get_HO_coords(atlas)
    plotting.plot_connectome(correlation_matrix, coords,
                         edge_threshold="95%", colorbar=True)

    plotting.show()
    view = plotting.view_connectome(correlation_matrix, coords, edge_threshold='90%')
    view.open_in_browser()

def graph_analysis(Adj):
    G = nx.from_numpy_array(Adj)
    # print(G)
    pos = nx.shell_layout(G)
    # nx.draw(G, pos = pos)
    # plt.show()
    global_efficieny = nx.global_efficiency(G)
    partitions = community.louvain_communities(G)
    # print(partitions)
    modularity = community.modularity(G, partitions)
    L = np.nan_to_num(nx.normalized_laplacian_matrix(G).todense().A)
    sorted_eigvals = np.sort(np.linalg.eigvals(L))
    # plt.plot(sorted_eigvals)
    # plt.show()
    for eig in sorted_eigvals:
        if eig > sorted_eigvals[0]:
            fd_val = eig
            break
        else: 
            fd_val = sorted_eigvals[0]
    print(global_efficieny, modularity, fd_val)
    return [global_efficieny, modularity, fd_val]

######################################Not Main#############################




##################################### MAIN ################################



# sliceSizeStep = 1
# kernel_size = 5
# windowSize = 40
# fwhm = 10
# sampleFile = "TD_Standardized/1_standardized.nii.gz"
# corrFolder = "TD_Standardized"
# # sampleFile = "raw_fMRI_raw_bold.nii.gz"
# pixelDim = 2 #For now have fixed the pixel dimension as 3
# # getSamplePixelDim(sampleFile)
# kernel = generateKernel(kernel_size,fwhm,pixelDim)
# convolvedThing = convolveWithKernel(kernel,windowSize)
# plt.plot(convolvedThing)
# plt.show()
# convolvedThing = convolvedThing.reshape((1,convolvedThing.shape[0]))
# [timeseriesMatrix,atlas,labels] = getParcellations(sampleFile)
# copyThing = copyConvolvedThing(convolvedThing,timeseriesMatrix.shape[1])
# windowThings = getWindows(timeseriesMatrix,copyThing,convolvedThing.shape[1],sliceSizeStep)
# windowThings = np.nan_to_num(windowThings)
# # correlation_matrices = measuringCorrelation(windowThings)



# plotCorrelationMatrix(correlation_matrices[:,:,0],atlas,labels[1:])
# print(correlation_matrices.shape)
# plt.imshow(windowThings[:,:,0])
# plt.show()


# analysis for one window
# total_windows = windowThings.shape[2] 
# i = 0
# efficiencies = np.ndarray(total_windows)#windowThings.shape[2])
# modularities = np.ndarray(total_windows)
# fd_values = np.ndarray(total_windows)
# for window_num in range(total_windows): #windowThings.shape[2]):
#     current_window = windowThings[:,:,window_num].T
#     ### trying to calculate sparse inverse covariance matrix
#     try:
#         [sparse_cov, sparse_prec] = group_sparse_covariance([current_window], 0.1)
#         connectivity_matrix = sparse_prec[:,:,0]
#         np.fill_diagonal(connectivity_matrix,0)
#         # plotCorrelationMatrix(connectivity_matrix,atlas,labels[1:])
#         [efficiencies[i], modularities[i], fd_values[i]]= graph_analysis(connectivity_matrix)
#         i = i+1
#     except Exception as e:
#         print(e)
#         continue
#     print(window_num)

# # print(efficiencies, modularities, fd_values)        
# plt.figure(1)
# plt.subplot(311)
# plt.plot(efficiencies)
# plt.subplot(312)
# plt.plot(modularities)
# plt.subplot(313)
# plt.plot(fd_values)
# plt.show()