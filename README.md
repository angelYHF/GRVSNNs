This code repository includes the source code for the paper about GRVSNNs, other Bayesian models (BLasso, BRR, BayesCpi), and LassoNet.

The experimental framework (GRVSNNs) is based on Python. however, the proposed model is implemented model can be implemented in Colab envrionment with some packages. The current version is implemented in Colab envrionment, if you choose the other environment, it should be easy to transfer also. 

**For the researcher who prefer to use R, we also make GRVSNNs_.py for the proposed GRVSNN model. You can call Python from R. The steps are as follows:

#install reticulate package provided by R

install.packages("reticulate") 

**steps for the anaconda environment setup:

Start menu --> Anaconda Promtopen and then run the environment setup as:

1. conda create -n tf_r_env python=3.9
2. conda activate tf_r_env
3. pip install tensorflow==2.10.0 numpy==1.24.4 pandas scikit-learn scipy bayesian-optimization
   
In R (or Rstudio)

1. library(reticulate)
2. use_condaenv("tf_r_env", required = TRUE)
3. py_config()
4. tf <- import("tensorflow")
5. print(tf$version)
6. activate tf using: 
conda activate tf_r_env

install packages needed in the code

pip install tensorflow==2.10.0 numpy==1.24.4 pandas scikit-learn scipy bayesian-optimization

in R (or Rstudio) using the code below:

install packages(“reticulate”)
library(reticulate)
use_condaenv("tf_r_env", required = TRUE)

#import the python module
grvsn <- import("GRVSNNs_")


#run the pipeline (here mice data is the example)

mse_result <- grvsn$run_training_pipeline(
  loadings_path = "C:/GRVSNN/data/loadings.csv",
  micedata_path = "C:/GRVSNN/data/micedata.csv"
)

print(mse_result)


For the Bayes Methods: BLasso, BayesCpi, BRR.

For the code Bayes Methods, there are several Bayes methods exclude three methods we showed above. There are also BayesB, BayesC methods. So please check the code carefully what kind of method you want to use. Because there are some calculations about loadings, there are different kinds of checks that need to do.


This repo includes the pedigree information for different datasets that we used in the paper. Due to the big size (especially for the pig data), we only provide the basic matrix of the pedigree info. You can read them as a matrix to do loadings calculation first, and then combine the results with the genomic markers to do the multi-task predictions. Please notice that we have given some discription about the parameters selection, but it also has smart initialization and other tricks for your own data. 


Requirements:

The code for the GRVSNN uses the following Python packages and they are required: pandas, numpy, sklearn, tensorflow, and bayes_opt. 

And the code for Bayesian methods, please remember to install the packages BGLR, and dplyr. 



Usage:

The current code is using the Colab environment, you can download the data accordingly and run the code step by step. Or you can run the code using your own Python environment.

Contact:
For any question, please can contact angelfyh@gmail.com

Citation:

It is coming ....
