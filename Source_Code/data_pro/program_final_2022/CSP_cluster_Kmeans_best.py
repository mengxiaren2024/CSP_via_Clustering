import json
import pymongo
import pandas as pd
from functools import reduce
from pandas import Series,DataFrame
from collections import Counter
from lxml import etree
from lxml import html
from scipy import stats
from sklearn import manifold
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
fetch_dire=["child-src", "connect-src", "default-src", "font-src", "frame-src", "img-src", "manifest-src",
                    "media-src", "object-src", "prefetch-src", "script-src", "style-src", "worker-src","style-src-elem",
                    "script-src-elem", "style-src-attr", "script-src-attr"]
special_process_schemes=["wss","ws","http","https","data","blob"]
customized_scheme="customized_scheme"
file_path="/Volumes/Elements/CSP_HOMEPAGE/"
csp_concat_vec_set=[]

file_name_con = file_path+"/df_merge_csp.csv"
csv_data = pd.read_csv(file_name_con, low_memory=False)
#######concat CSP vec
df_csp_vec_all = pd.DataFrame(csv_data)
df_csp_vec_all.to_csv((file_path+"/df_all_features.csv"))
df_csp_vec_all=df_csp_vec_all.drop(columns=['Unnamed: 0'])
drop_col_set=[]
col_zero=[]
col_sum=[]
for col in df_csp_vec_all.columns:
    #if ( col.find("_ex_do")==-1 and df_csp_vec_all[col].sum() < 10):
    if(df_csp_vec_all[col].sum() == 0):
        drop_col_set.append(col)
        col_zero.append(col)
    elif(df_csp_vec_all[col].sum() == len(df_csp_vec_all)):
        drop_col_set.append(col)
        col_sum.append(col)
# print("zero col:")
# for zc in col_zero:
#     print(zc)
# print("sum col:")
# for sc in col_sum:
#     print(sc)

df_csp_vec_all = df_csp_vec_all.drop(columns=drop_col_set)
for index,row in df_csp_vec_all.iterrows():
    col_num=0
    for col in df_csp_vec_all.columns:
        if(col!="Site" and col!="csp_con" and row[col]==1):
            col_num=1
    if col_num==1:
        df_csp_vec_all=df_csp_vec_all.drop(index=index)
df_csp_vec_all.to_csv(file_path+"row_zero_site.csv")

file_name_con = file_path + "/result-3/label_results_kmeans_elkan.csv"
cluster_10= pd.read_csv(file_name_con, low_memory=False)
#cluster_10=cluster_10.drop(columns=["lab2","lab3","lab4","lab5","lab6","lab7","lab8","lab9","lab10","lab11","lab14","lab13","lab15"])
###########csp template:
cluster_num=0
for lab in range(0,12):
    cluster_template=[]
    cluster=cluster_10[cluster_10["lab12"]==lab]
    cluster=cluster.drop(columns="lab12")
    for col in cluster:
        if((cluster[col].sum()>0) and col!="Unnamed: 0"):
        # if(col!="Unnamed: 0" and (cluster[col].sum()>(len(cluster)/2) or cluster[col].sum()==(len(cluster)/2))):
        #if (cluster[col].sum() >0):
            cluster_template.append("& "+col.replace("_"," ")+" &"+str(cluster[col].sum()))
    #print("cluster_"+str(cluster_num)+":("+str(len(cluster))+") (half number:"+str((len(cluster)/2))+")")
    print("\\multirow{"+str(len(cluster_template))+"}{*}{cluster"+str(cluster_num)+"(size:"+str(len(cluster))+")}")
    for tem in cluster_template:
        print(tem)
    cluster_num=cluster_num+1
    print("########################################################################")
#
fetch_dire=["child-src", "connect-src", "default-src", "font-src", "frame-src", "img-src", "manifest-src",
                    "media-src", "object-src", "prefetch-src", "script-src", "style-src", "worker-src","style-src-elem",
                    "script-src-elem", "style-src-attr", "script-src-attr"]
cluster_num=7
cluster_index=[]
cluster_index_1=[]
cluster_index_2=[]
cluster=cluster_10[cluster_10["lab12"]==cluster_num]
cluster=cluster.drop(columns="lab12")
row_max=0
columns=cluster.columns
columns=columns.drop("Unnamed: 0")
print(columns)
for index, row in cluster.iterrows():
 row_sum=0
 cols = []
 for col in columns:
    if(row[col]==1):
      cols.append(col)
 if(len(cols)==1 and ("frame-ancestors_self" in cols)):
  cluster_index_2.append(index)
 # if(row_sum==0):
 #  cluster_index.append(index)
 # if(row_sum==1):
 #  cluster_index_1.append(index)
     # if(row_sum>row_max):
     #  row_max=row_sum
  #mindex=index
print("row max:",row_max)
#print("mindex:",mindex)
print("cluster index:",cluster_index)
print(len(cluster_index))
print("cluster index 1:",cluster_index_1)
print(len(cluster_index_1))
print(cluster_index_2)
print(len(list(set(cluster_index_2))))
# for index, row in cluster.iterrows():
#  row_sum=0
#  for col in columns:
#     if(row[col]==1):
#      row_sum=row_sum+1
#  if(row_sum==0):
#   cluster_index.append(index)
#  if(row_sum==1):
#      for col in columns:
#          if(row[col]==1):
#              cluster_index_1.append(index)
#  if(row_sum>1):
#      cluster_index_2.append(index)
#      if(row_sum>row_max):
#       row_max=row_sum
#       mindex=index
# print("row max:",row_max)
# print("mindex:",mindex)
# print("cluster index:",cluster_index)
# print(len(cluster_index))
# print("cluster index 1:",cluster_index_1)
# print(len(cluster_index_1))
# print(cluster_index_2)
# print(len(cluster_index_2))
#
# #
# # for index, row in cluster.iterrows():
# #  if(index in cluster_index):
# #   for col in columns:
# #     if(row[col]==1):
# #         print("1 col is:",index," ",col)
# #
# # print("#############cluster_index:")
# # print(cluster_index_2)
# # print(len(cluster_index_2))
# #
# # for index, row in cluster.iterrows():
# #  if(index in cluster_index_2):
# #   for col in columns:
# #     if(row[col]==1):
# #         print("2 col is:",index," ",col)
#
#
# # cluster_num=12
# # cluster_index=[]
# # cluster_index0=[]
# # cluster_index_2=[]
# # cluster=cluster_10[cluster_10["lab14"]==cluster_num]
# # cluster=cluster.drop(columns="lab14")
# # row_max=0
# # columns=cluster.columns
# # columns=columns.drop("Unnamed: 0")
# # for index, row in cluster.iterrows():
# #  fetch_sum=0
# #  no_fetch=0
# #  for col in columns:
# #     if(row[col]>0):
# #         fetch_flag = 0
# #         for dire in fetch_dire:
# #           if(col.find(dire)>-1):
# #              fetch_sum=fetch_sum+1
# #              fetch_flag=1
# #         if(fetch_flag==0):
# #             no_fetch=no_fetch+1
# #  if( fetch_sum>0 and no_fetch==0):
# #   cluster_index.append(index)
# #  if(no_fetch>0 and  fetch_sum>0):
# #   cluster_index_2.append(index)
# # print(cluster_index)
# # print(len(cluster_index))
# # #
# # #
# # for index, row in cluster.iterrows():
# #  if(index in cluster_index):
# #   for col in columns:
# #     if(row[col]==1):
# #         fetch_flag=0
# #         for dire in fetch_dire:
# #             if(col.find(dire)>-1):
# #                 fetch_flag=1
# #         if(fetch_flag==0):
# #            print("1 col is:",index," ",col)
# #
# #
# # print("#############cluster_index:")
# # print(cluster_index_2)
# # print(len(cluster_index_2))
# # for index, row in cluster.iterrows():
# #  if(index in cluster_index_2):
# #   for col in columns:
# #     if(row[col]==1):
# #         fetch_flag=0
# #         for dire in fetch_dire:
# #             if(col.find(dire)>-1):
# #                 fetch_flag=1
# #         if(fetch_flag==0):
# #            print("1 col is:",index," ",col)
#
# # i=0
# # for index, row in cluster.iterrows():
# #  if(index in cluster_index_2):
# #     if(row["block-all-mixed-content"]==0):
# #         for col in columns:
# #          if (row[col] == 1):
# #             fetch_flag = 0
# #             for dire in fetch_dire:
# #                 if (col.find(dire) > -1):
# #                     fetch_flag = 1
# #             if (fetch_flag == 0):
# #                 print("2 col is:", index, " ", col)
#
# # for index, row in cluster.iterrows():
# #  de_sum=0
# #  no_de=0
# #  for col in columns:
# #     if(col.split("_")[0]=="default-src" and ((col.split("_")[1] in special_process_schemes) or (col.split("_")[1] in customized_scheme)) and row[col]>0):
# #     #if (row[col] > 0 and col.find("script-src_unsafe-inline")>-1 and ):
# #  #if (row[col] > 0 and col.split("_")[0]=="default-src"  and ((col.split("_")[1] in special_process_schemes) or (col.split("_")[1]==customized_scheme))):
# #         de_sum=de_sum+1
# #     if (col.find("default-src")==-1 and row[col] > 0):
# #         no_de=no_de+1
# #  if(de_sum>0):
# #   cluster_index.append(index)
# #  if(de_sum==0):
# #   cluster_index_2.append(index)
# # print(cluster_index)
# # print(len(cluster_index))
# #
# # for index, row in cluster.iterrows():
# #  if(index in cluster_index):
# #   for col in columns:
# #       if ( row[col] > 0):
# #           print("1 col is:",index," ",col)
# #           #print(index)
# #
# # print("#############cluster_index:")
# # print(cluster_index_2)
# # print(len(cluster_index_2))
# # for index, row in cluster.iterrows():
# #  if(index in cluster_index_2):
# #   for col in columns:
# #       if (row[col] > 0):
# #           print("2 col is:",index," ",col)
#
#
#
# # cluster_num=1
# # cluster_index=[]
# # cluster_index0=[]
# # cluster_index_2=[]
# # cluster_index_3=[]
# # cluster=cluster_10[cluster_10["lab12"]==cluster_num]
# # cluster=cluster.drop(columns="lab12")
# # row_max=0
# # columns=cluster.columns
# # columns=columns.drop("Unnamed: 0")
# # for index, row in cluster.iterrows():
# #  unline=0
# #  uneval=0
# #
# #  if(row["script-src_unsafe-inline"]==1):
# #         unline=1
# #  if (row["script-src_unsafe-eval"]==1):
# #         uneval = 1
# #  if( unline>0 and uneval>0):
# #   cluster_index.append(index)
# #  if(unline>0 and  uneval==0):
# #   cluster_index_2.append(index)
# #  if(unline==0 and  uneval>0):
# #   cluster_index_3.append(index)
# # print(cluster_index)
# # print(len(cluster_index))
# # for index, row in cluster.iterrows():
# #  if(index in cluster_index):
# #       if (row["script-src_unsafe-inline"] > 0):
# #           print("2 col is:",index," script-src_unsafe-inline")
# #       if (row["script-src_unsafe-eval"] > 0):
# #           print("2 col is:",index," script-src_unsafe-eval")
# # print(cluster_index_2)
# # print(len(cluster_index_2))
# # print(cluster_index_3)
# # print(len(cluster_index_3))
#
# cluster_num=11
# cluster_index=[]
# cluster_index0=[]
# cluster_index_2=[]
# cluster_index_3=[]
# cluster=cluster_10[cluster_10["lab12"]==cluster_num]
# cluster=cluster.drop(columns="lab12")
# row_max=0
# columns=cluster.columns
# columns=columns.drop("Unnamed: 0")
# for index, row in cluster.iterrows():
#  default=0
#  for col in columns:
#   if(row[col]==1):
#       #for dire in fetch_dire:
#           if(  col.find("script-src_http")>-1):
#              default=1
#  if( default>0):
#   cluster_index.append(index)
#  if(default==0):
#   cluster_index_2.append(index)
# print("fetch:",cluster_index)
# print(len(cluster_index))
# for index, row in cluster.iterrows():
#  if(index in cluster_index):
#      for col in columns:
#       if(row[col]==1 and col.find("script-src")>-1):
#           print("1 col is:",index," ",col)
# print(cluster_index_2)
# print(len(cluster_index_2))
# print(cluster_index_3)
# print(len(cluster_index_3))

#coverage calculation
file_name_con = "/Volumes/Elements/CSPNEW_DATA/CSP_CLUSTER/df_all_csp_coverage.csv"
csv_data = pd.read_csv(file_name_con, low_memory=False)
df_csp_all = pd.DataFrame(csv_data)
df_csp_all=df_csp_all.drop(columns=['Unnamed: 0'])
for lab in range(0,12):
    cluster=df_csp_all[df_csp_all["lab12"]==lab]
    print("###########################################")
    print("cluster ", str(lab+1))
    print("coverage:",cluster["coverage"].sum())
    print("len:",len(cluster["coverage"]))
    print("percentage:",float((cluster["coverage"].sum()/len(cluster["coverage"]))*100))

