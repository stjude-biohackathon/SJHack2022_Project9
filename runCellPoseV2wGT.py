##------------------------------------------------------------------------------
'''
Run CellPose2 w helper functions and segmentation performance.
-Requires ground truth masks, identical to OG image, with _Mask.<imagetype> appended.
-Requires Cellpose conda env (see: https://cellpose.readthedocs.io/en/latest/index.html)

NOTE: User supplied ground truth Class 1 values = 0
NOTE: Cellpose generated segmentation Class 1 values != 0 

Run Examples (from jupyter console): 

!python runCellPoseV2wGT.py /home/zhanggrp/jwilli16/jwilli16_cellpose/IHC_Test_Set /home/zhanggrp/jwilli16/jwilli16_cellpose/cellpose_output_vanilla

pretrained_model=‘/research/sharedresources/cbi/public/data/hackathon_data/cellpose_models/cellpose_residual_on_style_on_concatenation_off_IHC_annotated_cp_2022_04_11_14_18_03.575595’

!python runCellPoseV2wGT.py /home/zhanggrp/jwilli16/jwilli16_cellpose/IHC_Test_Set /home/zhanggrp/jwilli16/jwilli16_cellpose/cellpose_output_trained -FT 1 -PT -1 -PTM /research/sharedresources/cbi/public/data/hackathon_data/cellpose_models/cellpose_residual_on_style_on_concatenation_off_IHC_annotated_cp_2022_04_11_14_18_03.575595

!python runCellPoseV2wGT.py /home/zhanggrp/jwilli16/jwilli16_cellpose/IHC_Test_Set /home/zhanggrp/jwilli16/jwilli16_cellpose/cellpose_output_vanilla_newparams -FT 1 -PT -1

Output:
1) Cellpose default plot w segmentation and flow (png)
2) F1 score metrics (tab delim)
3) raw seg masks (low intensity, png) 

'''
##------------------------------------------------------------------------------

# standard libraries
import os
import pandas as pd
import re
import sys
import time
import warnings
import glob
import argparse

# processing libraries
import numpy as np
import xarray as xr
from tqdm import tqdm
from skimage import io
from skimage.morphology import remove_small_objects
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# cellpose specific
from cellpose import models, utils, plot, transforms, metrics

#------------------------------------------------------------------------------
def parseArguments(): 
    parser = argparse.ArgumentParser()   
    #--- mandatory arguments ------
    parser.add_argument("InDir", 
        help="Directory containing all input images.", type=str)
    parser.add_argument("OutDir", 
        help="Directory containing all output results.", type=str)
    #--- default values ------
    parser.add_argument("-PTM", default=None,
        help="cellpose pretrained model, default=None", type=None)
    parser.add_argument("-M", default="TN1",
        help="cellpose model, default=TN1", type=str)
    parser.add_argument("-D", default=30,
        help="diameter default=30", type=int)
    parser.add_argument("-do_3D", default=False,
        help="dimension=3d, default=False", type=bool)
    parser.add_argument("-FT", default=0.4,
        help="Flow threshold", type=float)
    parser.add_argument("-PT", default=0.0,
        help="Probability threshold", type=float)    
    parser.add_argument("-IT", default="*.tif",
        help="Image type regex, default=*.tif", type=str)
    parser.add_argument("-A", default=None,
        help="Anisotropy, default=None", type=None) 
    parser.add_argument("-C", default=[0,0],
        help="Channels, default=[0,0]", type=list)     

    args = parser.parse_args()
    return args
        
#--------------------------------------------------------------------------------
def sampleWalk(InputDir, FileType):
    print("Retrieving sample paths")
    RegexPath = os.path.join(InputDir,FileType)
    Samples = glob.glob(RegexPath)
    Samples = [s for s in Samples if "Mask" not in s]
    print(f'{len(Samples)} images were found:')
    print(np.array(Samples))
    return Samples

#--------------------------------------------------------------------------------
def createPlots(ResultsDict, OutDir):
    print("Save Cellpose performance plots")
    for SampleName in ResultsDict.keys():
        # plot name
        PlotSample = ".".join(SampleName.split(".")[:-1])
        CellPosePlotPath = os.path.join(OutDir, "CellPose_"+PlotSample+".png")
        MaskPlotPath = os.path.join(OutDir, "Pred_"+PlotSample+".png")
        # extract results
        SampleData = ResultsDict[SampleName]
        Mask = SampleData[0].astype(np.uint16)
        Flow = SampleData[1][0].astype(np.uint16)
        Style = SampleData[2].astype(np.uint16)
        OGImage = SampleData[3].astype(np.uint16)
        GTImage = SampleData[4].astype(np.uint16)

        # plot cellpose data
        Fig = plt.figure(figsize=(12,5))
        plot.show_segmentation(Fig, OGImage, Mask, Flow)#, channels=channels[idx])
        plt.tight_layout()
        Fig.savefig(CellPosePlotPath)
        plt.clf()
        
        # plot pred mask/segments only
        io.imsave(MaskPlotPath, Mask)
        
#--------------------------------------------------------------------------------        
def runCellPose2(SamplePaths, ParameterDict, Model):
    print("Run model")
    ResultsDict = dict()
    for SamplePath in SamplePaths:
        # ground truth path
        ImageType = SamplePath.split(".")[-1]
        GroundTruthPath = ".".join(SamplePath.split(".")[:-1])+"_Mask."+ImageType
        #capture sample name
        SampleName = SamplePath.split("/")[-1]
        print("\t Running Cellpose on:",SampleName)
        #read sample
        Image = io.imread(SamplePath)
        GroundTruthMask = io.imread(GroundTruthPath)[:,:,0]
        #Image = addSpacing(Image)
        #Image = [rescaleIntensity(Image)]
        Masks, Flows, Styles = Model.eval(Image, **ParameterDict)
        ResultsDict[SampleName] = [Masks, Flows, Styles, Image, GroundTruthMask]   
    return ResultsDict

#--------------------------------------------------------------------------------
def buildParameters(LineArgs):
    print("Build parameters for model...")
    RunModelParamDict = dict(channels=LineArgs.C,
                        diameter=LineArgs.D,
                        flow_threshold=LineArgs.FT,
                        cellprob_threshold=LineArgs.PT,
                        do_3D=LineArgs.do_3D,
                        anisotropy=LineArgs.A)
    return RunModelParamDict

#--------------------------------------------------------------------------------
def prepMetrics(ImArray, Type):
    if Type == "GroundTruth":
        B = ImArray.flatten() == 0
    elif Type == "Pred":
        B = ImArray.flatten() != 0
    return B

#--------------------------------------------------------------------------------
def getMetrics(ResultsDict, OutDir):
    print("Running metrics vs ground truth...")
    ResultsList = []
    for SampleName in ResultsDict.keys():
        print(SampleName)
        # extract results
        SampleData = ResultsDict[SampleName]
        Mask = prepMetrics(SampleData[0].astype(np.uint16),"Pred")
        GroundTruth = prepMetrics(SampleData[4].astype(np.uint16),"GroundTruth")
        F1 = f1_score(GroundTruth, Mask)
        Results = [SampleName, F1]
        ResultsList.append(Results)
    # output path
    MetricsOutPath = os.path.join(OutDir, "Metrics.txt")
    ResultsDF = pd.DataFrame(ResultsList, columns=["Sample","F1"])   
    ResultsDF.to_csv(MetricsOutPath, sep="\t", header=True,index=False)
    print(ResultsDF)
    
#--------------------------------------------------------------------------------
def main(LineArgs):
    # Input parameters
    ParameterDict = buildParameters(LineArgs)
    #--------
    # load model
    if LineArgs.PTM != None:
        Model = models.CellposeModel(gpu=True,pretrained_model=LineArgs.PTM)
    else:
        Model = models.CellposeModel(gpu=True,model_type=LineArgs.M)
    # return all sample image paths
    SamplePaths = sampleWalk(LineArgs.InDir, LineArgs.IT)
    # run model over sample list
    ResultsDict = runCellPose2(SamplePaths, ParameterDict, Model)
    # report metrics:
    getMetrics(ResultsDict, LineArgs.OutDir)
    # plot these results:
    createPlots(ResultsDict, LineArgs.OutDir)
    print("Pipeline completed!")

#------------------------------------------------------------------------------
if __name__ == '__main__':
    # Parse the arguments
    LineArgs = parseArguments()
    main(LineArgs)