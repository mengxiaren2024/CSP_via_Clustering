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

file_path="/Volumes/Elements/CSP_HOMEPAGE/"

csv_data_1 = pd.read_csv(file_path+"df_merge_csp.csv", low_memory=False)
df_csp_vec_all= pd.DataFrame(csv_data_1)
df_csp_vec_all=df_csp_vec_all.drop(columns=['Unnamed: 0',"domain","Site","csp_con"])
print("columns length:", df_csp_vec_all.columns," ***************************************")
drop_col_set=[]
col_zero=[]
col_sum=[]
for col in df_csp_vec_all.columns:
    print(col)
    #if ( col.find("_ex_do")==-1 and df_csp_vec_all[col].sum() < 10):
    if(df_csp_vec_all[col].sum() == 0):
        drop_col_set.append(col)
        col_zero.append(col)
    elif(df_csp_vec_all[col].sum() == len(df_csp_vec_all)):
        drop_col_set.append(col)
        col_sum.append(col)
print("zero col:")
for zc in col_zero:
    print(zc)
print("sum col:")
for sc in col_sum:
    print(sc)
df_csp_vec_all = df_csp_vec_all.drop(columns=drop_col_set)
print("delete colms:",drop_col_set)
print(df_csp_vec_all.columns," ",len(df_csp_vec_all.columns))
df_csp_vec_all.to_csv(file_path+"csp_vec_all.csv")
keep_col=[]
for col in df_csp_vec_all.columns:
    print("keep col:",col)
    col=col.split("_")[0]
    keep_col.append(col)
print("keep len:",len(list(set(keep_col))))
keep_col_2= list(set(keep_col))
all_directives = ["child-src", "connect-src", "default-src", "font-src", "frame-src", "img-src", "manifest-src",
                    "media-src", "object-src", "prefetch-src", "script-src", "style-src", "worker-src","style-src-elem",
                    "script-src-elem", "style-src-attr", "script-src-attr",
                    "base-uri","sandbox",
                    "form-action","frame-ancestors","navigate-to",
                    "block-all-mixed-content","upgrade-insecure-requests",
                    "trusted-types","require-sri-for", "require-trusted-types-for","plugin-types"#,"policy-definition" #additional directives
                  ]
for c in all_directives:
 if(c not in keep_col_2):
  print("keep_col:",c)

#########cluster CSP
label_all=[]
cspvec=np.array(df_csp_vec_all)
clf = KMeans(n_clusters=12,init = "k-means++",algorithm="elkan")
label=clf.fit_predict(cspvec)
print("eyebow score:",clf.inertia_)
if(len(np.unique(label))>1):
    print("num:", 12 , " clusters:", len(np.unique(label))," cluster eval:", metrics.silhouette_score(cspvec, label))
for i in  np.unique(label):
    print("cluster_",i,": ", len(cspvec[label == i]))
label_all.append(label)
# # ###########################################################################3d figure
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
cspv = tsne.fit_transform(cspvec)
df=np.hstack((cspv,label.reshape(-1,1)))


fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
u_labels = np.unique(label)
print(len(u_labels))
color=["red","blue","purple","darkblue","green","orange","#D12B60","black","brown","deeppink","olive", "lawngreen","darkorange","forestgreen","royalblue","navy","teal","chocolate","gold"]
for i in u_labels:
  ax.scatter(df[label == i, 0],df[label == i, 1],0, s = 40 , color = color[i], label = i)
        #ax.scatter(cspvec[label == i, 0],cspvec[label == i, 1],cspvec[label == i, 2], s = 40 , color = color[i], label = i)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.legend()
  fil2=file_path+"Kmeans_12_elkan.png"
  #fil2 = file_path + "Kmeans_" + str(cn) + "_elkan.png"
  plt.savefig(fil2)
i=12
for lab in label_all:
 col="lab"+str(i)
 df_csp_vec_all[col]=lab
 i=i+1
file1=file_path+"/label_results_kmeans_elkan.csv"
#file1=file_path+"/label_results_kmeans_elkan.csv"
df_csp_vec_all.to_csv(file1)

