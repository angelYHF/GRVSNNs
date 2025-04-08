This code repository includes the source code for the paper about GRVSNNs, other Bayesian models (BLasso, BRR, BayesCpi), and LassoNet.
The experimental framework (GRVSNNs) is based on Python. however, the proposed model is implemented model can be implemented in Colab envrionment with some packages. The current version is implemented in Colab envrionment, if you choose the other environment, it should be easy to transfer also. I will provide the R version for this GRVSNNs quite soon.

This repo includes the pedigree information for different datasets that we used in the paper. Due to the big size (especially for the pig data), we only provide the basic matrix of the pedigree info. You can read them as a matrix to do loadings calculation first, and then combine the results with the genomic markers to do the multi-task predictions. Please notice that we have given some discription about the parameters selection, but it also has smart initialization and other tricks for your own data. 

Requirements:
The code uses the following Python packages and they are required: pandas, numpy, sklearn, tensorflow, and bayes_opt. 

Usage
The current code is using the Colab environment, you can download the data accordingly and run the code step by step. Or you can run the code using your own Python environment.

Contact
For any question, you can contact angelfyh@gmail.com

Citation
It is coming ....
