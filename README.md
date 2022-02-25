# HeartSR
##  Joint Motion Correction and Super Resolution for Cardiac Segmentation via Latent Optimisation

###  1. pre-trained generative models of HR segmentations
 - pre-trained beta-VAE models are available in ./models/betaVAE
 - HR segmentations are available on [1] 
 - models are trained on HR segmentations with voxel size 1.2 x 1.2 x 2 mm, cropped and centered the mass center of cardiac mask
 
###  2. Demo on simulated degradation
 - Sim_demo.ipynb uses the data in ./data/HR_demo
 - Paired HR-LR segmentations are used for comparison

###  3. Demo on UK Biobank data
 - UKB_demo.ipynb uses the data in ./data/UKB_demo
 - data has already been cropped centered the cardiac mass center 
 - remember to adjust the image oreintation and make the crop operation when deal with new dataset

[1] Savioli, Nicolo; de Marvao, Antonio; O'Regan, Declan (2021), “Cardiac super-resolution label maps”, Mendeley Data, V1, doi: 10.17632/pw87p286yx.1
