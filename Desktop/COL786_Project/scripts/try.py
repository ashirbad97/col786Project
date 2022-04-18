import numpy as np
import nibabel
from nilearn import datasets
from nilearn.maskers import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting

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
    return convolvedThing

def getParcellations(sampleFile):
    # Retrieve the atlas and the data
    atlas = datasets.fetch_atlas_msdl()
    # Loading atlas image stored in 'maps'
    atlas_filename = atlas['maps']
    # Loading atlas data stored in 'labels'
    labels = atlas['labels']
    # Load the functional datasets
    # data = datasets.fetch_development_fmri(n_subjects=1)
    data = sampleFile
    print('First subject resting-state nifti image (4D) is located at: %s' %
        data)
    # Extract the time series
    masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True,
                            memory='nilearn_cache', verbose=5)
    masker.fit(data)
    time_series = masker.transform(data)
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
    windowedView = np.ndarray((timeseriesMatrixT.shape[0],windowSize, (timeseriesMatrixT.shape[1]-windowSize+sliceSizeStep)//3))
    for windowStart in range(0,timeseriesMatrixT.shape[1],sliceSizeStep):
        windowEnd = windowStart + windowSize
        if windowEnd < timeseriesMatrixT.shape[1]:
            # print(windowStart, windowEnd)
            sliceThing = timeseriesMatrixT[:,windowStart:windowEnd]
            np.append(windowedView, sliceThing)
    # print(windowedView.shape)
    return windowedView

def measuringCorrelation(windowThings):
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform(windowThings)[0]
    return correlation_matrix

def plotCorrelationMatrix(correlation_matrix,atlas,labels):
    np.fill_diagonal(correlation_matrix, 0)
    plotting.plot_matrix(correlation_matrix,  colorbar=True,
                     vmax=0.8, vmin=-0.8)
    coords = atlas.region_coords
    plotting.plot_connectome(correlation_matrix, coords,
                         edge_threshold="80%", colorbar=True)

    plotting.show()
    view = plotting.view_connectome(correlation_matrix, coords, edge_threshold='80%')
    view.open_in_browser()


sliceSizeStep = 3
kernel_size = 4
windowSize = 22
fwhm = 3
sampleFile = "TD_Standardized/0_standardized.nii.gz"
# sampleFile = "raw_fMRI_raw_bold.nii.gz"
pixelDim = 3 #For now have fixed the pixel dimension as 3
# getSamplePixelDim(sampleFile)
kernel = generateKernel(kernel_size,fwhm,pixelDim)
convolvedThing = convolveWithKernel(kernel,windowSize)
convolvedThing = convolvedThing.reshape((1,convolvedThing.shape[0]))
[timeseriesMatrix,atlas,labels] = getParcellations(sampleFile)
copyThing = copyConvolvedThing(convolvedThing,timeseriesMatrix.shape[1])
windowThings = getWindows(timeseriesMatrix,copyThing,windowSize,sliceSizeStep)
windowThings = np.nan_to_num(windowThings)
correlation_matrix = measuringCorrelation(windowThings)
plotCorrelationMatrix(correlation_matrix,atlas,labels)


#### 
def graph_analysis(Adj):
    G = nx.from_numpy_array(Adj)
    # print(G)
    pos = nx.shell_layout(G)
    # nx.draw(G, pos = pos)
    # plt.show()
    global_efficieny = nx.global_efficiency(G)
    partitions = community.best_partitions(G)
    modularity = community.modularity(partitions, G)
    sorted_eigvals = np.sort(np.linalg.eigvals(nx.normalized_laplacian_matrix(G).A))
    
    # print('Global Efficiency of graph: ', global_efficieny)
    return global_efficieny