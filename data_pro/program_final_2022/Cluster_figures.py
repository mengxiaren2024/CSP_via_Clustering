import json
import pymongo
import pandas as pd
from functools import reduce
from pandas import Series,DataFrame
from collections import Counter
from lxml import etree
from lxml import html
from scipy import stats
from sklearn import metrics
###################################################################LOGS
import numpy as np
from io import StringIO
from bs4 import BeautifulSoup
import seaborn as sns
import numpy as np
import tldextract
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pyclustering.utils.metric import distance_metric, type_metric
from pandas.io.json import json_normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
file_path="/Volumes/Elements/CSP_HOMEPAGE/"
file_name_con = file_path + "result-3/label_results_kmeans_elkan.csv"
clusters= pd.read_csv(file_name_con, low_memory=False)
clusters=clusters.drop(columns=["Unnamed: 0"])
cluster_whole=clusters.drop(columns="lab12")
print(cluster_whole.columns)
statistic_whole=dict.fromkeys(cluster_whole.columns,0)
set1=[]
for el in cluster_whole.columns:
    set1.append(el.split("_")[0])
print("find:",len(list(set(set1))))

for index,row in cluster_whole.iterrows():
    for col in cluster_whole.columns:
        if(row[col]==1):
            statistic_whole[col]=statistic_whole[col]+1
sorted_statistic_whole=sorted(statistic_whole.items(),key=lambda item: item[1],reverse=True)
sorted_features=[]
sorted_features_index=[]
sorted_feature_value=[]
i=0
for l in sorted_statistic_whole:
    sorted_features.append(l[0])
    sorted_features_index.append(i)
    sorted_feature_value.append((l[1]/len(cluster_whole))*100)
    i=i+1
for i in range(0,len(sorted_features)):
    print(sorted_features_index[i]+1," ",sorted_features[i]," ",sorted_feature_value[i])


# i=0
# for fea in sorted_features:
#      i=i+1
#      if((i % 3)!=0):
#          print("&",str(i), "&\\textcolor{blue}{",fea,"}")
#      else:
#          print("&", str(i), "&\\textcolor{blue}{",fea,"} hline")


# insecure_directive_index=[4,5,6,15,16,21,51,
#                           59,85,87,89,96,100,102,
#                           120,121,124,129,133,
#                           135,138,142,144,145,146,147,148,149,151,152,161,162,
#                           166,172,173,175,177,182,187,188,
#                           197,198,204,223,226,239,
#                           252,258,261,262,263,265]
#
# black_directive_index=[3,12,54,
#                           58,77,81,86,101,104,108,
#                           111,113,116,117,119,125,131,
#                           134,137,139,150,156,160,164,
#                           167,171,179,180,183,185,190,191,192,
#                           201,203,205,206,207,217,225,229,237,240,242,
#                           249,257,266,267,269,274,275]

insecure_directive_index=[4,5,6,15,17,21,51,61,85,87,95,99,103,
                          119,121,124,129,135,138,145,146,147,
                          151,152,159,160,164,169,170,172,176,181,186,
                          187,196,197,204,219,222,239,252,258,261,262,263,265]

black_directive_index=[3,12,54,
                          58,76,80,86,101,105,108,111,
                          113,115,116,117,118,125,
                          131,133,134,137,139,143,144,
                          148,149,150,156,158,162,165,
                          168,178,179,182,184,189,190,191,
                          201,203,205,206,207,213,221,225,237,240,242,
                          249,257,266,267,269,274,275]
print("insecure length:",len(insecure_directive_index))
print("black length:",len(black_directive_index))

secure_directive_index=[]
for i in sorted_features_index:
    if(((i+1) not in insecure_directive_index) and ((i+1) not in black_directive_index)):
        secure_directive_index.append(i+1)
print("secure length:",len(secure_directive_index))
print(secure_directive_index)


# for i in range(1,len(sorted_feature_value)+1):
#     if(i in secure_directive_index):
#         p1=plt.bar(i,sorted_feature_value[i-1] ,color="blue",width=1)
#     if (i in insecure_directive_index):
#         # if(sorted_feature_value[i-1]>20):
#         #     print("features:",sorted_features[i-1])
#         p2=plt.bar(i, sorted_feature_value[i-1], color="red",width=1)
#     if (i in black_directive_index):
#         p3=plt.bar(i, sorted_feature_value[i-1], color="black",width=1)
# plt.legend([p1, p2,p3], ["safe directive", "unsafe directive","uncertain directive"], loc='upper right')
# plt.xlabel('Feature Index')
# plt.ylabel('Popularity Percentage %')
# plt.title('Popularity of Features')
# file_name=file_path+"/statistic_whole.png"
# plt.savefig(file_name)
# plt.cla()


# i=0
# for fea in sorted_features:
#      i=i+1
#      if (i in secure_directive_index):
#          print("&",str(i), "&\\textcolor{blue}{",fea,"}")
#      if (i in insecure_directive_index):
#          print("&", str(i), "&\\textcolor{red}{", fea,"}")
#      if (i in black_directive_index):
#          print("&",str(i), "&\\textcolor{black}{",fea,"}")

for lab in range(0,12):
    s_cluster = dict.fromkeys(sorted_features,0)
    subcluster=clusters[clusters["lab12"]==lab]
    subcluster=subcluster.drop(columns="lab12")
    for index,row in subcluster.iterrows():
        for col in subcluster.columns:
            if(row[col]==1):
                s_cluster[col]=s_cluster[col]+1
    for col in subcluster.columns:
        s_cluster[col]=(s_cluster[col]/(len(subcluster)))*100
    #print("original col val:", s_cluster)
    pl10=0
    pm10_20=0
    pm20_50=0
    pm50=0
    pf=0
    p100=0
    cindex=0
    safe_col=0
    unsafe_col=0
    unknown_col=0
    col_list=[]
    for col in sorted_features:
        cindex=cindex+1
        if(s_cluster[col]>90):
            pm50=pm50+1
            col_new=col.split("_")[0]
            col_list.append(col_new)
            print("pop_col:",col)
            if (cindex in secure_directive_index and s_cluster[col]>0):
                safe_col = safe_col + 1
                print("safe_pop_col:", col)
            if (cindex in insecure_directive_index and s_cluster[col]>0):
                unsafe_col = unsafe_col + 1
                print("unsafe_pop_col:", col)
            if (cindex in black_directive_index and s_cluster[col]>0):
               unknown_col = unknown_col + 1
               print("unknown_pop_col:", col)
        if (s_cluster[col] > 0):
            pf = pf + 1
        if((s_cluster[col]>0 and s_cluster[col]<10) or s_cluster[col]==10):
            pl10=pl10+1
        if (((s_cluster[col] > 10) and (s_cluster[col] < 20)) or s_cluster[col]==20):
            pm10_20=pm10_20+1
        if ((s_cluster[col] > 20) and (s_cluster[col] < 50 or s_cluster[col]==50)):
            pm20_50 = pm20_50+1
        if ( s_cluster[col]==100):
            p100 = p100+1
    print("cluster", lab+1)
    print("number of data points:",len(subcluster))
    print("total number of features:",pf)
    print("features no more than 10%:",pl10)
    print("features 10-20%:",pm10_20)
    print("features 20-50%:",pm20_50)
    print("features more than 90%:",pm50)
    print("features 100%:", p100)
    print("features safe col:", safe_col)
    print("features unsafe col:", unsafe_col)
    print("features unknown col:", unknown_col)
    print("nunber of differnt directives:",len(list(set(col_list))))
    print("col:",list(set(col_list)))
    print("#########################################################################")

    # for i in range(1, len(sorted_feature_value) + 1):
    #     if (i in secure_directive_index):
    #        p1= plt.bar(i, s_cluster[sorted_features[i-1]],  color="blue",width=1)
    #     if (i in insecure_directive_index):
    #        p2= plt.bar(i, s_cluster[sorted_features[i - 1]], color="red",width=1)
    #     if (i in black_directive_index):
    #        p3= plt.bar(i, s_cluster[sorted_features[i - 1]], color="black",width=1)
    # #plt.plot(sorted_features_index, s_cluster.values(), markersize=2)
    # plt.axis([0,275,0,100])
    # plt.xlabel('Feature Index')
    # plt.ylabel('Popularity Percentage %')
    # plt.title('Popularity of Features')
    # plt.legend([p1,p2,p3], ["safe directive", "unsafe directive","uncertain directive"], loc='upper right')
    # file_name = file_path + "/cluster_pop_feature_bar"+str(lab)+".png"
    # plt.savefig(file_name)
    # plt.cla()

