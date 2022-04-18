from os import listdir
from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as mvn

from try_rohit import *


def run(sampleFile):
    try:
        sliceSizeStep = 1
        kernel_size = 4
        windowSize = 40
        fwhm = 10
        sliceSizeStep = 1
        # sampleFile = "TD_Standardized/1_standardized.nii.gz"
        # sampleFile = "raw_fMRI_raw_bold.nii.gz"
        pixelDim = 2 #For now have fixed the pixel dimension as 3
        # getSamplePixelDim(sampleFile)
        kernel = generateKernel(kernel_size,fwhm,pixelDim)
        convolvedThing = convolveWithKernel(kernel,windowSize)
        convolvedThing = convolvedThing.reshape((1,convolvedThing.shape[0]))
        [timeseriesMatrix,atlas,labels] = getParcellations(sampleFile)
        copyThing = copyConvolvedThing(convolvedThing,timeseriesMatrix.shape[1])
        windowThings = getWindows(timeseriesMatrix,copyThing,convolvedThing.shape[1],sliceSizeStep)
        windowThings = np.nan_to_num(windowThings)
        correlation_matrices = measuringCorrelation(windowThings)
        # print("Shape of correlation matrix is : ",correlation_matrices.shape)
        return correlation_matrices
    except Exception as e:
        print(e)

def saveCorrMatrices(correlation_matrices,fileName):
    matFileName = "correlationMatrixmatFiles/"+str(fileName)+"_correlationMatrix.mat"
    scipy.io.savemat(matFileName,{'data':correlation_matrices})
    print("Saved : ",fileName)

# ENTRY POINT FUNCTION
def callSaveCorrMat():
    mahaList = []
    corrFolder = "ADHD_Standardized"
    fileList = os.listdir(corrFolder)
    for file in fileList:
        if file.endswith(".nii.gz"):
            correlation_matrices = run(os.path.join(corrFolder,file))
            saveCorrMatrices(correlation_matrices,os.path.join(corrFolder,file))
            print("Done for : ",os.path.join(corrFolder,file))

# Takes a given matrix file as input, disintegrates all the sliding windows and deforms them into single vectors and appends them into one giant
# matrix for each individual subject and returns it back
def violateTheMatrix(matFile):
    # Not sure why this shape !
    # Will contain the bundle of all the decomposed chota matrix in a stack
    badaWindow = np.ndarray((matFile.shape[2],matFile.shape[0]*matFile.shape[1]))
    # Since input is a 3D matrix
    # Iterate through all of the windows
    for window in range(matFile.shape[2]):
        # contains the 2D individual window
        chotaWindow = matFile[:,:,window]
        # decompose each window into a (1X(row*col)) 1D matrix
        chotaWindow = chotaWindow.reshape(1,matFile.shape[0]*matFile.shape[1])
        # Not sure why this shape !
        # Stack the chotaWindows over each other vertically
        badaWindow[window,:] = chotaWindow
    # Return the deformed windows of a subject
    return badaWindow

# Function to read the corelation matrix in .mat file
def readCorrelationMatrix(inputCorrelationMatrix):
    readMatFile = scipy.io.loadmat(inputCorrelationMatrix)['data']
    badaWindow = violateTheMatrix(readMatFile)
    # We get the shape of this matrix as (no.Of windows from input,dim[0]*dim[1] of each windows)
    return badaWindow

def saveMahaMatrix(mahaMatrix,phenotype):
    matFileName = str(phenotype)+"_groupCorrelationMatrix.mat"
    scipy.io.savemat(matFileName,{'data':mahaMatrix})
    print("Saved : ",matFileName)

# ENTRY POINT FUNCTION
# Reads through all of the saved matrices of the patients and parses them and saves into one big matrix of the particular phenotype
def parseCorrMatrix():
    phenotype = "ADHD_Standardized"
    matFolderPath = "correlationMatrixmatFiles/"+phenotype
    # Iterate through all of the files in the folder
    fileList = os.listdir(matFolderPath)
    # To get the shape of 1 matrix to create the shape of the maha matrix
    farziMatrix = scipy.io.loadmat(os.path.join(matFolderPath,fileList[0]))['data']
    print(farziMatrix.shape)
    # Create the mahamatrix of comp[uted row and column
    col = farziMatrix.shape[0]*farziMatrix.shape[1]
    # Not sure if the shape inititalization of mahaMatrix is correct or not !!!!!!!!!!!!!!!
    mahaMatrix = np.ndarray((0,col))
    for file in fileList:
        fullFilePath = os.path.join(matFolderPath,file)
        # Fetches all the deformed matrices of a given person and appends them into a huge group matrix and saves them
        badaWindow = readCorrelationMatrix(fullFilePath)
        mahaMatrix = np.concatenate((mahaMatrix,badaWindow),axis=0)
    # Although mahaMatrix shape is exactly that of what you instructed
    # save the maha matrix
    saveMahaMatrix(mahaMatrix,phenotype)

def joinGroupCorrMatrix():
    # Folder path where all the group corr matrices are present
    groupCorrMatrixDirPath = "groupCorrMatrix"
    corrMatrixFiles = os.listdir(groupCorrMatrixDirPath)
    # Coz I don't know anyother way to declare the size of the numpy matrix
    nakliMatrix = scipy.io.loadmat(os.path.join(groupCorrMatrixDirPath,corrMatrixFiles[0]))['data']
    megaMatrix = np.ndarray((0,nakliMatrix.shape[1]))
    for file in corrMatrixFiles:
        if file.endswith(".mat"):
            miniMatrix = scipy.io.loadmat(os.path.join(groupCorrMatrixDirPath,file))['data']
            megaMatrix = np.concatenate((megaMatrix,miniMatrix),axis=0)
    # Saves the mega matrix containing both the phenotypes
    mahaMatFileName = "combinedGroupsMatrix.mat"
    scipy.io.savemat(mahaMatFileName,{'data':megaMatrix})
    print("Saved : ",mahaMatFileName)

# Plot and Find the optimal k value for the combinedCorrMatFiles
def runKMeansOnGroupMatrices():
    inputGroupCorrMat = "groupCorrMatrix/TD_Standardized_groupCorrelationMatrix.mat"
    # inputGroupCorrMat = "groupCorrMatrix/combinedGroupsMatrix.mat"
    groupCorrMat = scipy.io.loadmat(inputGroupCorrMat)['data']
    groupCorrMat = np.nan_to_num(groupCorrMat)
    # bic = []
    # davies = []
    silh = []
    # cali = []
    for k in range(2,7):
        km = KMeans(n_clusters=k, random_state=37)
        km.fit(groupCorrMat)
        pred = km.predict(groupCorrMat)
        gmm = GaussianMixture(n_components=k, init_params='kmeans')
        gmm.fit(groupCorrMat)
        silh.append(silhouette_score(groupCorrMat, pred))
        # bic.append(gmm.bic(groupCorrMat))
        print('done for k = ', k)
    # plt.subplot(211)
    # plt.plot(bic)
    # plt.subplot(212)
    plt.plot(np.arange(2,7), silh)
    plt.xlabel('k value')
    plt.ylabel('Silhoutte score')
    plt.show()

# run optimal kmeans on single group and return centroid correlation matrices and corresponding dwell times
def runMainKMeansSingleGroup():
    optimalK = 4
    inputGroupCorrMat = "groupCorrMatrix/ADHD_Standardized_groupCorrelationMatrix.mat"
    # inputGroupCorrMat = "groupCorrMatrix/combinedGroupsMatrix.mat"
    groupCorrMat = scipy.io.loadmat(inputGroupCorrMat)['data']
    groupCorrMat = np.nan_to_num(groupCorrMat)
    km = KMeans(n_clusters=optimalK, random_state=37)
    km.fit(groupCorrMat)
    pred = km.predict(groupCorrMat)
    centroids = km.cluster_centers_
    # find dwell times 
    dwell_times = np.unique(pred, return_counts=True)
    num_ROIs = int(np.sqrt(centroids.shape[1]))
    print(num_ROIs)
    clutsers_as_corr_mat = np.ndarray((num_ROIs, num_ROIs, centroids.shape[0]))
    for cluster_num in range(centroids.shape[0]):
        centroid = centroids[cluster_num, :].reshape(num_ROIs, num_ROIs)
        clutsers_as_corr_mat[:,:,cluster_num] = centroid
    print(dwell_times)
    return [clutsers_as_corr_mat, dwell_times]

def runMainKMeansBothGroups():
    optimalK = 4
    ADHD_mat_file = "groupCorrMatrix/ADHD_Standardized_groupCorrelationMatrix.mat"
    ADHD_mat = scipy.io.loadmat(ADHD_mat_file)['data']
    ADHD_mat = np.nan_to_num(ADHD_mat)
    TD_mat_file = "groupCorrMatrix/TD_Standardized_groupCorrelationMatrix.mat"
    TD_mat = scipy.io.loadmat(TD_mat_file)['data']
    TD_mat = np.nan_to_num(TD_mat)
    inputGroupCorrMat = "groupCorrMatrix/combinedGroupsMatrix.mat"
    groupCorrMat = scipy.io.loadmat(inputGroupCorrMat)['data']
    groupCorrMat = np.nan_to_num(groupCorrMat)
    km = KMeans(n_clusters=optimalK, random_state=37)
    km.fit(groupCorrMat)
    pred_TD = km.predict(TD_mat)
    pred_ADHD = km.predict(ADHD_mat)
    centroids = km.cluster_centers_
    # find dwell times 
    TD_dwell_times = np.unique(pred_TD, return_counts=True)
    ADHD_dwell_times = np.unique(pred_ADHD, return_counts=True)
    num_ROIs = int(np.sqrt(centroids.shape[1]))
    # print(num_ROIs)
    clutsers_as_corr_mat = np.ndarray((num_ROIs, num_ROIs, centroids.shape[0]))
    for cluster_num in range(centroids.shape[0]):
        centroid = centroids[cluster_num, :].reshape(num_ROIs, num_ROIs)
        clutsers_as_corr_mat[:,:,cluster_num] = centroid
    return [clutsers_as_corr_mat, TD_dwell_times, ADHD_dwell_times]

def generateGroupGraphMetrics():
    # runKMeansOnGroupMatrices()
    # Retrieve the atlas and the data
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    # Loading atlas data stored in 'labels'
    labels = atlas.labels #atlas['labels']
    [cluster_correlation_matrices, TD_dwell_times, ADHD_dwell_times] = runMainKMeansBothGroups()
    print(TD_dwell_times, ADHD_dwell_times)
    cor = cluster_correlation_matrices[:,:,3]
    cor[cor < 0.3] = 0
    plotCorrelationMatrix(cor, atlas, labels[1:])
    for n in range(cluster_correlation_matrices.shape[2]):
        Adj = cluster_correlation_matrices[:,:,n]
        Adj[Adj<0.3] = 0
        graph_analysis(Adj)

# Performs the stats against each individual of the phenotype
def generateIndividualGraphMetrics():
    badaDwell = []
    # Reading the group correlation matrix
    optimalK = 4
    inputGroupCorrMat = "groupCorrMatrix/combinedGroupsMatrix.mat"
    groupCorrMat = scipy.io.loadmat(inputGroupCorrMat)['data']
    groupCorrMat = np.nan_to_num(groupCorrMat)
    km = KMeans(n_clusters=optimalK, random_state=37)
    km.fit(groupCorrMat)
    centroids = km.cluster_centers_
    num_ROIs = int(np.sqrt(centroids.shape[1]))
    # Loop through all of the individual correlation matrices
    phenotype = "TD_Standardized"
    matFolderPath = "correlationMatrixmatFiles/"+phenotype
    for matFile in listdir(matFolderPath):
        badaWindow = readCorrelationMatrix(os.path.join(matFolderPath,matFile))
        indvMat = np.nan_to_num(badaWindow)
        pred_ADHD = km.predict(indvMat)
    # find dwell times 
        _,dwell_times = np.unique(pred_ADHD, return_counts=True) 
        print(_,dwell_times)
    #     badaDwell.append(dwell_times)
    # print(badaDwell)
    clutsers_as_corr_mat = np.ndarray((num_ROIs, num_ROIs, centroids.shape[0]))
    for cluster_num in range(centroids.shape[0]):
        centroid = centroids[cluster_num, :].reshape(num_ROIs, num_ROIs)
        clutsers_as_corr_mat[:,:,cluster_num] = centroid
    return [clutsers_as_corr_mat, dwell_times, dwell_times]


# def staticConnectivity():
#     standardFolder = "ADHD_Standardized"
#     fileList = os.listdir(standardFolder)
#     for file in fileList:
#         if file.endswith(".nii.gz"):
#             correlation_matrices = run(os.path.join(corrFolder,file))
#             saveCorrMatrices(correlation_matrices,os.path.join(corrFolder,file))
#             print("Done for : ",os.path.join(corrFolder,file))

def main():
    # Function to read the standardized files and save them into a correlation matrix for each individual subject data present inside the dir
    # callSaveCorrMat()
    # Function to read all the correlation matrices present in a dir and combines them to form the group matrix for all the subjects
    # parseCorrMatrix()
    # Function to parse two group matrix and combine them into one
    # joinGroupCorrMatrix()
    # Function to run kMeans operation on the group matrices
    generateGroupGraphMetrics()
    # Function to perform static connectivity
    # staticConnectivity()
    # Function to run kMeans on the individual matrices
    # generateIndividualGraphMetrics()
    # runMainKMeansSingleGroup()

if __name__ == "__main__":
    # Calls the main function only if executed this file from the cli
    main()