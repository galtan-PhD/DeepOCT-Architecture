# DeepOCT-Architecture



<div>
The script helps to train and test DeepOCT architecture. <br>
The DeepOCT is an optical coherence tomography analysis architecture, that is based on Convolutional neural network, to perform identification of Diabetic Macular Edema (DME) against nonDME using the detailed analysis capabilities of Deep Learning.  
  
  <br>
  <br>

The paper of the project is under review. Therefore, the repository has limitations in to be used a complete analysis. <br>
  <b>The current version of the repository is consists of </b>
  
  <ol>
    <li>Constant files for DeepOCT parameters (<b>constants.py</b>) </li>
      <li>A limited version of preprocessing procedure (<b>data_prep.py</b>) </li>
      <li>Keras implementation for DeepOCT architecture (<b>model.py</b>)</li>
      <li>Training procedure with classification parameters (<b>train.py</b>)</li>
      <li>Test file that prepares a raw OCT image for DeepOCT (<b>test.py</b>) </li>
   </ol>


  <b>*** </b> The paper is in review process, the complete preprocessing steps and optimum weight of DeepOCT <b>('deepOCTmodel.h5')</b> will be added to the repository after a publication. <br>
  <b>*** </b> The optimum weight of DeepOCT <b>('deepOCTmodel.h5')</b> will be shared as (<b>TransferLearning.py</b>) to be used in <b>Transfer Learning </b> as the pre-trained weights for different problems. 
  
  



Please cite this article :<br>
 <b> Gokhan ALTAN, DeepOCT: An Optimized and Pruned Deep Learning Architecture to Analyze Macular Edema on OCT images (Under Review) </b>

  <br><br>
  
  # DeepOCT-GradCam Visualizations for randomly selected OCT images from ZhangLab
  ## DeepOCT on localization of very severe DME
  ![DeepOCT-DME-Large](http://www.gokhanaltan.com/wp-content/uploads/OCT_DME_Large.png)
  ## DeepOCT on localization of nonDME
  ![DeepOCT-DME-nonDME](http://www.gokhanaltan.com/wp-content/uploads/OCT_Normal.png)
  ## DeepOCT on localization of very severe DME
  ![DeepOCT-DME-Small](http://www.gokhanaltan.com/wp-content/uploads/OCT_DME_Small.png)
  
  

 

</div>
