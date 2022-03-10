import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
from sklearn import metrics
from sklearn_extra.cluster import KMedoids
# from sklearn.metrics import pairwise_distances, silhouette_samples, silhouette_score, davies_bouldin_score
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tslearn.metrics import dtw_path
import time
from statsmodels.tsa.stattools import acf, pacf
from statistics import mean, median, stdev, pstdev

start_time = time.time()
# FORMATTING ########################################################################
def load_data(data_file): #load in the data 
# Data file should have normalized solar, load and wind data in columns 6:8; Year, month, date in columns 1:3 (or just month data in column 2)
    data_train = pd.read_excel(data_file)  
    x_load = (data_train.iloc[:, [6,7,8,1,2,3]]).values
    x_train=x_load
    return x_train
# x_train=load_data("train-8-1yr.xlsx")     #data for 1 year starting 4/1/2018
# x_train=load_data("train-9-6mo.xlsx")     #data for 6 months starting 4/1/2018
x_train=load_data("train-23-April-24mo.xlsx")     #data for 2 years starting 4/1/2018
# x_train=load_data("Train-20-Oct18-24mo.xlsx")     #data for 2 years starting 10/1/2018
# x_train=load_data("Train-21-Oct18-12mo.xlsx")     #data for 1 year starting 10/1/2018
# x_train=load_data("train-25-April-24mo-wkday.xlsx")     #data for 2 years - only weekdays starting 4/2/2018
# x_train=load_data("train-25-April-24mo-wkend_rev2.xlsx")     #data for 2 years - only weekends starting 4/1/2018
# x_train=load_data("train-26-April-24mo_shoulder.xlsx")     #data for 2 years - only shoulder seasons starting 4/1/2018
# x_train=load_data("train-26-April-24mo_summer.xlsx")     #data for 2 years - only summer seasons from 6/20 to 9/18 (inclusive)
# x_train=load_data("train-26-April-24mo_winter.xlsx")     #data for 2 years - only winter seasons from 11/16 to 2/14 (inclusive)
print("--- %s seconds LOAD ---" % (time.time() - start_time))
# Data formatting. (full_mat is 365x24x3 - not used), each of ls_mat, (lw_mat and sw_mat - not used) is 365x24x2; 
def format_data(x_train): #separate the days into rows#
    length=int(len(x_train)/24)
    solar_mat=np.zeros((length,24))
    load_mat=np.zeros((length,24))
    month_mat=np.zeros((length,24))
    ls_mat = []
    # IN the case here, only load-solar data are used, but load+solar+wind or any other two-ple combination can be used
    for i in range(0,length):
        solar_mat[i,:]=x_train[i*24:i*24+24,0]
        load_mat[i,:]=x_train[i*24:i*24+24,1]
        month_mat[i,:]=x_train[i*24:i*24+24,4]
        months=month_mat[:,0]
        b = [load_mat[i],solar_mat[i]] # b is the list format for faster computation for full_mat, add wind_mat
        ls_mat.append(b)
    return(solar_mat,load_mat, ls_mat, months)
solar_mat,load_mat,ls_mat, month_list = format_data(x_train)
print("--- %s seconds FORMAT ---" % (time.time() - start_time))

# DTW ##############################################################################
length=int(len(x_train)/24)
# The following is unconstrained DTW
def manual_DTW(p,q):
    n, m = len(p), len(q)
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n):
        dtw_matrix[i,-1] = np.inf
    for j in range(m):
        dtw_matrix[-1, j] = np.inf
    dtw_matrix[-1, -1] = 0
    
    for i in range(n):
        for j in range(m):
            cost = np.linalg.norm(p[i] - q[j])**2
            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return np.sqrt(dtw_matrix[n-1,m-1])

full_d_mat = np.zeros((length,length))
# Compute dtw matrix by calculating DTW distance for each pair
for i in range(length):
    for j in range(length):
        full_d_mat[i][j] = manual_DTW(ls_mat[i],ls_mat[j]) #ls_mat can be replaced by a different multi-dimensional matrix
print("--- %s seconds DTW ---" % (time.time() - start_time))
# Write the DTW matrix output to Excel file
pd.DataFrame(full_d_mat).to_excel(excel_writer = "DTW_Matrix.xlsx")

# Flattening of ls-mat for flat clustering (KMeans, KMedoids)
ls_mat_array = np.array(ls_mat)
ls_mat_2D = ls_mat_array.reshape(length, -1) 
# Metric generates Separation Score(SS) and Cohesion Score(CS)
def metric(ls_mat, ls_rep, t, length, clust_vals, clust_counts):
    new_metric_1 = np.zeros((length, t)) # DTW Distance between each day and different cluster reps
    sort_nm = np.zeros((length, t))
    new_metric_2 = np.zeros((length,7)) # 6 metrics
    for i in range(length):
        for j in range(t):
            new_metric_1[i][j] = manual_DTW(ls_mat[i],ls_rep[j])
        new_metric_2[i,0] = new_metric_1[i, clust_vals[i]-1] # Distance to own cluster mean for each day (MA_i)
        new_metric_2[i,1] = min(new_metric_1[i]) # Minimum distance from a day to any cluster mean (MB_i)
        new_metric_2[i,2] = new_metric_2[i,0] -new_metric_2[i,1] # Difference - 0 is good, positive is bad
        sort_nm[i] = np.sort(new_metric_1[i]) # Sort for low to high of distances to all cluster means
        #  compute lowest of distances from a day to other cluster means (excluding its own)
        if new_metric_2[i,2] ==0:
            new_metric_2[i,3] = sort_nm[i, 1] # Take the second lowest number in the sorted array if the difference is 0 (clustering is "good") (MC_i)
        else:
            new_metric_2[i,3] = sort_nm[i, 0] # Take the lowest number if the difference is not 0. (MC_i)
        new_metric_2[i,4] = new_metric_2[i,3] - new_metric_2[i,1] # Positive Difference indicates good clustering (MC_i - MB_i)
        new_metric_2[i,5] = new_metric_2[i,3] - new_metric_2[i,0] # (x_i = MC_i - MA_i)
        new_metric_2[i,6] = new_metric_2[i,5]/ max(new_metric_2[i,3], new_metric_2[i,0])# Similar to Silhouette but at individual points
    count10 = np.mean(new_metric_2[:,5]) # SS = mean(x_i)
    clust_dist_tot = np.zeros((t,1))
    clust_dist = np.zeros((t,4))
    for j in range(t):
        clust_dist[j,0] = 10
    for i in range(length): # Loop to compute the total, maximum of MA_i
        clust_dist_tot[clust_vals[i]-1] = clust_dist_tot[clust_vals[i]-1] + new_metric_2[i,0]
        clust_dist[clust_vals[i]-1, 1] = max(clust_dist[clust_vals[i]-1, 1], new_metric_2[i,0]) # Maximum distance in a cluster between representation and the members
    count4 = max(clust_dist[:,1]) # CS = Max(Max (MA_i))  
    counts_Ave = [count4,  count10]
    return (counts_Ave, new_metric_1, new_metric_2, clust_dist)


# CLUSTERING #######################################################################
# Specify the cluster number range over which the metrics need to be evaluated
# Final plots are for the highest cluster number (tend -1)
# Clustering is done using KMeans (KM prefix), KMedoids (KMed), AHC with complete linkage (Comp) and AHC with average linkage (Ave)
tstart = 2
tend = 21
Comp_Met = np.zeros((tend-tstart, 6)) # Metrics for Complete linkage
Ave_Met = np.zeros((tend-tstart, 6))
KM_Met = np.zeros((tend-tstart, 6))
KMed_Met = np.zeros((tend-tstart, 6))
Clust1 = np.zeros((length,tend-tstart)) 
KM_Clust1 = np.zeros((length,tend-tstart))
Ave_Clust1 = np.zeros((length,tend-tstart))
KMed_Clust1 = np.zeros((length,tend-tstart))
Clust2 = np.zeros((tend,tend-tstart))
# Loop for the full range of cluster values to generate metrics data
for t1 in range (tstart,tend):
    KMed_sk_clust1=KMedoids(n_clusters=t1).fit(ls_mat_2D) # Cluster flattened data using KMedoids
    KM_sk_clust1=sklearn.cluster.KMeans(n_clusters=t1).fit(ls_mat_2D) # Cluster flattened data using KMeans
    # Following lined are AHC with complete and average linkage, respectively, precomputed DTW distance is used
    sk_clust1=sklearn.cluster.AgglomerativeClustering(n_clusters=t1, affinity='precomputed', 
                                                  memory=None, connectivity=None, compute_full_tree='auto',linkage='complete').fit(full_d_mat)
    Ave_sk_clust1=sklearn.cluster.AgglomerativeClustering(n_clusters=t1, affinity='precomputed', 
                                                  memory=None, connectivity=None, compute_full_tree='auto',linkage='average').fit(full_d_mat)
    # Comput the traditional metrics and store as below for all types of clustering
    # Complete Linkage
    SMet1 = metrics.silhouette_score(full_d_mat,sk_clust1.labels_, metric= "precomputed")
    DMet1 = metrics.davies_bouldin_score(full_d_mat,sk_clust1.labels_)
    CHMet1 = metrics.calinski_harabasz_score(full_d_mat,sk_clust1.labels_)
    Comp_Met[t1-tstart, 0:4] = [t1,CHMet1, DMet1,SMet1]
    # Average Linkage
    Ave_SMet1 = metrics.silhouette_score(full_d_mat,Ave_sk_clust1.labels_, metric= "precomputed")
    Ave_DMet1 = metrics.davies_bouldin_score(full_d_mat,Ave_sk_clust1.labels_)
    Ave_CHMet1 = metrics.calinski_harabasz_score(full_d_mat,Ave_sk_clust1.labels_)
    Ave_Met[t1-tstart, 0:4] = [t1,Ave_CHMet1, Ave_DMet1,Ave_SMet1]
    # KMeans
    KM_SMet1 = metrics.silhouette_score(ls_mat_2D,KM_sk_clust1.labels_)
    KM_DMet1 = metrics.davies_bouldin_score(ls_mat_2D,KM_sk_clust1.labels_)
    KM_CHMet1 = metrics.calinski_harabasz_score(ls_mat_2D,KM_sk_clust1.labels_)
    KM_Met[t1-tstart, 0:4] = [t1,KM_CHMet1, KM_DMet1,KM_SMet1]
    # KMedoids
    KMed_SMet1 = metrics.silhouette_score(ls_mat_2D,KMed_sk_clust1.labels_)
    KMed_DMet1 = metrics.davies_bouldin_score(ls_mat_2D,KMed_sk_clust1.labels_)
    KMed_CHMet1 = metrics.calinski_harabasz_score(ls_mat_2D,KMed_sk_clust1.labels_)
    KMed_Met[t1-tstart, 0:4] = [t1,KMed_CHMet1, KMed_DMet1,KMed_SMet1]
    # Further data extraction from each type of clustering
     # Complete Linkage
    full_sk_vals=1+sk_clust1.labels_             #assign the labels by dayindex to an array (labels from sklearn start at 0, so add 1 to each) size: nx1, content: 1:t
    sk_count_temp = np.bincount(full_sk_vals)   #get the number of occurrences for each label (i.e each cluster). size: (t+1)x1 
    num_clust = np.nonzero(sk_count_temp)[0]    #get length t
    sk_cluster_counts=sk_count_temp[num_clust]  #take the last t clusters since we dont use 0.
     # Average Linkage
    Ave_full_sk_vals=1+Ave_sk_clust1.labels_             #assign the labels by dayindex to an array (labels from sklearn start at 0, so add 1 to each) size: nx1, content: 1:t
    Ave_sk_count_temp = np.bincount(Ave_full_sk_vals)   #get the number of occurrences for each label (i.e each cluster). size: (t+1)x1 
    Ave_num_clust = np.nonzero(Ave_sk_count_temp)[0]    #get length t
    Ave_sk_cluster_counts=Ave_sk_count_temp[Ave_num_clust]  #take the last t clusters since we dont use 0.
    # KMeans
    KM_full_sk_vals=1+KM_sk_clust1.labels_             #assign the labels by dayindex to an array (labels from sklearn start at 0, so add 1 to each) size: nx1, content: 1:t
    KM_sk_count_temp = np.bincount(KM_full_sk_vals)   #get the number of occurrences for each label (i.e each cluster). size: (t+1)x1 
    KM_num_clust = np.nonzero(KM_sk_count_temp)[0]    #get length t
    KM_sk_cluster_counts=KM_sk_count_temp[KM_num_clust]  #take the last t clusters since we dont use 0.
    # KMedoids
    KMed_full_sk_vals=1+KMed_sk_clust1.labels_             #assign the labels by dayindex to an array (labels from sklearn start at 0, so add 1 to each) size: nx1, content: 1:t
    KMed_sk_count_temp = np.bincount(KMed_full_sk_vals)   #get the number of occurrences for each label (i.e each cluster). size: (t+1)x1 
    KMed_num_clust = np.nonzero(KMed_sk_count_temp)[0]    #get length t
    KMed_sk_cluster_counts=KMed_sk_count_temp[KMed_num_clust]  #take the last t clusters since we dont use 0.
    
    t = len(set(full_sk_vals))
    Clust1[:,t1-tstart] = full_sk_vals
    KM_Clust1[:,t1-tstart] = KM_full_sk_vals
    Ave_Clust1[:,t1-tstart] = Ave_full_sk_vals
    KMed_Clust1[:,t1-tstart] = KMed_full_sk_vals
    month_count_by_cluster=np.zeros((length,t))
# 
    solar_rep = np.zeros((t,24))
    load_rep = np.zeros((t,24))
    ls_rep = []
    Ave_solar_rep = np.zeros((t,24))
    Ave_load_rep = np.zeros((t,24))
    Ave_ls_rep = []
    KM_solar_rep = np.zeros((t,24))
    KM_load_rep = np.zeros((t,24))
    KM_ls_rep = []
    KMed_solar_rep = np.zeros((t,24))
    KMed_load_rep = np.zeros((t,24))
    KMed_ls_rep = []
# Calculate Representative day by computing median for each type of clustering
# Complete Linkage
    for j in range (t):
        temp1 = []
        temp2 = []
        for i in range (length):
            if j == full_sk_vals[i]-1:
                temp1.append(solar_mat[i])
                temp2.append(load_mat[i])
        solar_rep[j] = np.median(temp1, axis=0)
        load_rep[j] = np.median(temp2, axis=0)
        ls_rep.append([load_rep[j], solar_rep[j]])
# Average Linkage
    for j in range (t):
        temp1 = []
        temp2 = []
        for i in range (length):
            if j == Ave_full_sk_vals[i]-1:
                temp1.append(solar_mat[i])
                temp2.append(load_mat[i])
        Ave_solar_rep[j] = np.median(temp1, axis=0)
        Ave_load_rep[j] = np.median(temp2, axis=0)
        Ave_ls_rep.append([Ave_load_rep[j], Ave_solar_rep[j]])
# KMeans
    for j in range (t):
        temp1 = []
        temp2 = []
        for i in range (length):
            if j == KM_full_sk_vals[i]-1:
                temp1.append(solar_mat[i])
                temp2.append(load_mat[i])
        KM_solar_rep[j] = np.median(temp1, axis=0)
        KM_load_rep[j] = np.median(temp2, axis=0)
        KM_ls_rep.append([KM_load_rep[j], KM_solar_rep[j]])
# KMedoids
    for j in range (t):
        temp1 = []
        temp2 = []
        for i in range (length):
            if j == KMed_full_sk_vals[i]-1:
                temp1.append(solar_mat[i])
                temp2.append(load_mat[i])
        KMed_solar_rep[j] = np.median(temp1, axis=0)
        KMed_load_rep[j] = np.median(temp2, axis=0)
        KMed_ls_rep.append([KMed_load_rep[j], KMed_solar_rep[j]])

# Generate the new metrics (SS and CS) and add to the conventional metrics file *_Met

    counts_Ave, new_metric_1, new_metric_2, clust_dist = metric(ls_mat, ls_rep, t, length, full_sk_vals, sk_cluster_counts)
    Comp_Met[t1-tstart, 4:6]= [ counts_Ave[0], counts_Ave[1]]
    Ave_counts_Ave, Ave_new_metric_1, Ave_new_metric_2, Ave_clust_dist = metric(ls_mat, Ave_ls_rep, t, length, Ave_full_sk_vals, Ave_sk_cluster_counts)
    Ave_Met[t1-tstart, 4:6]= [Ave_counts_Ave[0], Ave_counts_Ave[1]]
    KM_counts_Ave, KM_new_metric_1, KM_new_metric_2, KM_clust_dist = metric( ls_mat, KM_ls_rep, t, length, KM_full_sk_vals, KM_sk_cluster_counts)
    KM_Met[t1-tstart, 4:6]= [KM_counts_Ave[0], KM_counts_Ave[1]]
    KMed_counts_Ave, KMed_new_metric_1,  KMed_new_metric_2, KMed_clust_dist = metric(ls_mat, KMed_ls_rep, t, length, KMed_full_sk_vals, KMed_sk_cluster_counts)
    KMed_Met[t1-tstart, 4:6]= [KMed_counts_Ave[0], KMed_counts_Ave[1]]
# *_Clust1 is a listing of each day and its cluster assignment for each type of clustering - saved in Excel
with pd.ExcelWriter('clusters1.xlsx') as writer:  
    pd.DataFrame(Clust1).to_excel(writer, sheet_name='Complete DTW')
    pd.DataFrame(Ave_Clust1).to_excel(writer, sheet_name='Average DTW')
    pd.DataFrame(KM_Clust1).to_excel(writer, sheet_name='KMeans')
    pd.DataFrame(KMed_Clust1).to_excel(writer, sheet_name='KMedoids')
# 
#    
print("--- %s seconds CLUSTER ---" % (time.time() - start_time))
# # *_Metrics is a summary of metrics for each type of clustering
with pd.ExcelWriter('Metrics.xlsx') as writer:  
    Metrics = pd.DataFrame(Comp_Met)
    Metrics.columns = ["Clusters", "CH", "DB", "Silhouette", "CS",  "SS"]
    Metrics.to_excel(writer, sheet_name='Complete')
    Ave_Metrics = pd.DataFrame(Ave_Met)
    Ave_Metrics.columns = ["Clusters", "CH", "DB", "Silhouette",  "CS", "SS"]
    Ave_Metrics.to_excel(writer, sheet_name='Average')
    KM_Metrics = pd.DataFrame(KM_Met)
    KM_Metrics.columns = ["Clusters", "CH", "DB", "Silhouette",  "CS",  "SS"]
    KM_Metrics.to_excel(writer, sheet_name='KMeans')
    KMed_Metrics = pd.DataFrame(KMed_Met)
    KMed_Metrics.columns = ["Clusters", "CH", "DB", "Silhouette", "CS", "SS"]
    KMed_Metrics.to_excel(writer, sheet_name='KMedoids')
# pd.DataFrame(new_metric_1).to_excel(excel_writer = "NewMetrics1_DTW.xlsx")


#RESULTS AND PLOTTING##############################################################
length=int(len(x_train)/24)
# Define labels and colors for plotting - upto 36 clusters covered below, add more if necessary
def labels_and_colors(cluster_vals): #labels and colors for the plotting
    cluster_string=list(map(str,cluster_vals))
    
    cluster_colors = ['cornflowerblue' if wd == "1"  \
                      else 'red' if wd=="2" \
                      else 'coral' if wd=="3" \
                      else 'violet' if wd=="4" \
                      else 'orange' if wd=="5" \
                      else 'brown' if wd=="6" \
                      else 'cyan' if wd=="7" \
                      else 'pink' if wd=="8" \
                      else 'lightgreen' if wd=="9" \
                      else 'olive' if wd=="10" \
                      else 'green' if wd=="11" \
                      else 'purple' if wd=="12" \
                      else 'blue' if wd=="13" \
                      else 'red' if wd=="14" \
                      else 'yellowgreen' if wd=="15" \
                      else 'aqua' if wd=="16" \
                      else 'darkorange' if wd=="17" \
                      else 'lime' if wd=="18" \
                      else 'orchid' if wd=="19" \
                      else 'chocolate' if wd=="20" \
                      else 'skyblue' if wd=="21" \
                      else 'palegreen' if wd=="22" \
                      else 'crimson' if wd=="23" \
                      else 'teal' if wd=="24" \
                      else 'plum' if wd=="25" \
                      else 'tan' if wd=="26" \
                      else 'tomato' if wd=="27" \
                      else 'navy' if wd=="28" \
                      else 'lawngreen' if wd=="29" \
                      else 'magenta' if wd=="30" \
                      else 'beige' if wd=="31" \
                      else 'khaki' if wd=="32" \
                      else 'wheat' if wd=="33" \
                      else 'lavender' if wd=="34" \
                      else 'steelblue' if wd=="35" \
                      else 'magenta' if wd=="36" \
                      else wd for wd in cluster_string]
    cluster_labels=[]
    for i in range(0,length):
        cluster_labels.append("C" +cluster_string[i]+", Day "+str(i+1))
    return cluster_labels,cluster_colors    
# Plotting all plots   
def all_plot(load_mat,solar_mat, load_rep,solar_rep, full_cluster_colors,full_cluster_vals,f,clust_counts,month_list):
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    if (t % 2) == 0:
        tnew = t
    else:
        tnew = t+1
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    idx = [xin for xin in range(24)]            #initialize a bunch of variables
    mdx = [xin+1 for xin in range(12)]
    month_count_by_cluster=np.zeros((length,t))
    mcnz=[]
    plt.figure(f)
    for i in range(0,length):                   #plotting the actual data. Putting separate clusters on separate subplots.
        month_count_by_cluster[i,full_cluster_vals[i]-1]=month_list[i]   
        plt.subplot(6,tnew/2,full_cluster_vals[i])
        plt.plot(idx,load_mat[i].T,color=full_cluster_colors[i])
        plt.ylim((0,1))
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        ptitle="#"+str(full_cluster_vals[i])+" ("+str(clust_counts[full_cluster_vals[i]-1])+" days)"
        plt.title(label=ptitle)
        plt.ylabel('Load \n [norm]')
        plt.subplot(6,tnew/2,tnew+full_cluster_vals[i])
        plt.plot(idx,solar_mat[i].T,color=full_cluster_colors[i])
        plt.ylim((0,1))
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        # ptitle="#"+str(full_cluster_vals[i])+"\n #days="+str(clust_counts[full_cluster_vals[i]-1])
        # plt.title(label=ptitle)
        plt.ylabel('Solar \n [norm]')
       
    for tl in range(0,t):                       #this is for the histograms. # of occurances for each month. For each cluster.
        mcnz.append(month_count_by_cluster[:,tl][month_count_by_cluster[:,tl] != 0])    
    h = int(tnew/2)
    for j in range(0,h):                        #create the histograms at the bottom of the plot.
        data=mcnz[j]
        left_bin = 0.5
        right_bin = 13.5
        plt.subplot(6,tnew/2,j+1) 
        plt.plot(idx,load_rep[j],color='black',linewidth=2)
        plt.subplot(6,tnew/2,tnew+j+1) 
        plt.plot(idx,solar_rep[j],color='black',linewidth=2)
        plt.subplot(6,tnew/2,2*tnew+j+1)
        counts, bins, _ = plt.hist(data,np.arange(left_bin, right_bin), edgecolor='white', linewidth=2)
        counts = counts.astype(int)
        for n, b in zip(counts, bins):
            if n > 0:
                plt.gca().text(b, n, str(n))
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.xlabel('Month')
        mdx1 = [3,6,9,12]
        plt.xticks(mdx1, ( '3', '6','9', '12'))     
        # plt.subplot(4,t,j+1)                    #plotting these averages as thick black lines.
        # plt.plot(idx,load_represent[j],color='black',linewidth=4)
        # plt.subplot(4,t,t+j+1)
        # plt.plot(idx,solar_represent[j],color='black',linewidth=4)
        # plt.subplot(4,t,t*2+j+1)
        # plt.plot(idx,wind_represent[j],color='black',linewidth=4)      
    for j in range(h,t):                        #create the histograms at the bottom of the plot.
        data=mcnz[j]
        left_bin = 0.5
        right_bin = 13.5
        plt.subplot(6,tnew/2,j+1) 
        plt.plot(idx,load_rep[j],color='black',linewidth=2)
        plt.subplot(6,tnew/2,tnew+j+1) 
        plt.plot(idx,solar_rep[j],color='black',linewidth=2)
        plt.subplot(6,tnew/2,2*tnew+j+1)
        counts, bins, _ = plt.hist(data,np.arange(left_bin, right_bin), edgecolor='white', linewidth=2)
        counts = counts.astype(int)
        for n, b in zip(counts, bins):
            if n > 0:
                plt.gca().text(b, n, str(n))
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.xlabel('Month')
        plt.xticks(mdx1, ('3', '6', '9', '12'))
       
    #Just adding the axis names once the loops are done.    
    plt.subplot(6,h,1)
    plt.ylabel('Load[n]')
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_visible(True) 
    plt.subplot(6,h,h+1)
    plt.ylabel('Load [n]')
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_visible(True) 
    plt.subplot(6,h,2*h+1)
    plt.ylabel('Solar [n]')
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_visible(True) 
    plt.subplot(6,h,3*h+1)
    plt.ylabel('Solar [n]')
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_visible(True) 
    plt.subplot(6,h,4*h+1)
    plt.ylabel('# days')
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_visible(True)
    plt.subplot(6,h,5*h+1)
    plt.ylabel('# days')
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_visible(True)
    plt.subplots_adjust(wspace=0.05, hspace=0.45) #spacing the subplots
    return

full_sk_labels,full_sk_colors=labels_and_colors(full_sk_vals)       #get the labels and colors for plotting.
Ave_full_sk_labels,Ave_full_sk_colors=labels_and_colors(Ave_full_sk_vals)       #get the labels and colors for plotting.
KM_full_sk_labels,KM_full_sk_colors=labels_and_colors(KM_full_sk_vals)       #get the labels and colors for plotting.
KMed_full_sk_labels,KMed_full_sk_colors=labels_and_colors(KMed_full_sk_vals)       #get the labels and colors for plotting.
#  Select which plots to show from below by uncommenting - all 4 are available to plot
# all_plot(load_mat,solar_mat,load_rep,solar_rep, full_sk_colors,full_sk_vals,1,sk_cluster_counts,month_list)
all_plot(load_mat,solar_mat,Ave_load_rep,Ave_solar_rep, Ave_full_sk_colors,Ave_full_sk_vals,2,Ave_sk_cluster_counts,month_list)
# all_plot(load_mat,solar_mat,KM_load_rep,KM_solar_rep, KM_full_sk_colors,KM_full_sk_vals,3,KM_sk_cluster_counts,month_list)
# # all_plot(load_mat,solar_mat,wind_mat,KMed_load_rep,KMed_solar_rep, KMed_full_sk_colors,KMed_full_sk_vals,4,KMed_sk_cluster_counts,month_list)
print("--- %s seconds MAIN PLOTS---" % (time.time() - start_time))
plt.show()
