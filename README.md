This code repository includes the source code for the paper about GRVSNNs, other Bayesian models (BLasso, BRR, BayesCpi), and LassoNet.

The experimental framework (GRVSNNs) is based on Python. however, the proposed model is implemented model can be implemented in Colab envrionment with some packages. The current version is implemented in Colab envrionment, if you choose the other environment, it should be easy to transfer also. For the R, you can call Python from R. The steps are as follows:

install.packages("reticulate") #install reticulate package provided by R

library(reticulate)
os <- import("os")
os
os$listdir(".") #check files in the current path


#install packages needed 
py_install("pandas")
py_install("numpy")
py_install("bayesian-optimization")
py_install("tensorflow")
py_install("scikit-learn")

#path for GRVSNNs.py, loadings.py
source_python(" ")

#print






For the Bayes Methods: BLasso, BayesCpi, BRR.

For the code Bayes Methods, there are several Bayes methods exclude three methods we showed above. There are also BayesB, BayesC methods. So please check the code carefully what kind of method you want to use. Because there are some calculations about loadings, there are different kinds of checks that need to do.


This repo includes the pedigree information for different datasets that we used in the paper. Due to the big size (especially for the pig data), we only provide the basic matrix of the pedigree info. You can read them as a matrix to do loadings calculation first, and then combine the results with the genomic markers to do the multi-task predictions. Please notice that we have given some discription about the parameters selection, but it also has smart initialization and other tricks for your own data. 


Requirements:

The code for the GRVSNN uses the following Python packages and they are required: pandas, numpy, sklearn, tensorflow, and bayes_opt. 

And the code for Bayesian methods, please remember to install the packages BGLR, and dplyr. 



Usage:

The current code is using the Colab environment, you can download the data accordingly and run the code step by step. Or you can run the code using your own Python environment.

Contact:
For any question, you can contact angelfyh@gmail.com

Citation:

It is coming ....
