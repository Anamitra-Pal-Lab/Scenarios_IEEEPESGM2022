# Scenarios_IEEEPESGM2022
Representative Scenarios to Capture Renewable Generation Stochasticity and Cross-Correlations. 
This code creates correlated daily scenarios using different methods (KMeans, KMedoids, DTW with AHC) and also generates metrics for different cluster counts.
To run the program, import Excel file with format as intructed in the code. 
Data file should have normalized solar, load and wind data in columns 6:8; Year, month, date in columns 1:3 (or just month data in column 2).
The program outputs 3 Excel files (DTW_Matrix, clusters1 and Metrics) that have information about the clusters and metrics. 
See this link for the paper: https://arxiv.org/abs/2202.03588
