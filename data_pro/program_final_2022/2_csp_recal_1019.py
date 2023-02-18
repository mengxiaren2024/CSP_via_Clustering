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

#select enforcement csp homapes
file_path="/Volumes/Elements/CSP_HOMEPAGE/"
enforcement_homepage_total=[]
reportonly_homepage_total=[]
data_set_len=0
whole_csp_set=[]
for df in range(1,16):
    file_name_con = file_path+"/df_concat_c"+str(df)+".csv"
    csv_data = pd.read_csv(file_name_con, low_memory=False)
    df_con = pd.DataFrame(csv_data)
    df_con.drop_duplicates(["domain","cspmetaheader","cspheader","cspmetacon","cspcontent"], 'last', inplace=True)
    df_con["domain"]=df_con["domain"]+df*10000
    whole_csp_set.append(df_con)
    for index,row in df_con.iterrows():
        if(row["cspmetaheader"].lower()=="content-security-policy"or row["cspheader"].lower()=="content-security-policy"):
            enforcement_homepage_total.append(row["domain"])
    data_set_len=data_set_len+len(df_con[df_con["ctype"]=="CSP"])
print("dataset len:",data_set_len)
print(enforcement_homepage_total)
print("number of home page in enforcement mode:",len(list(set(enforcement_homepage_total))))



#select milticsp in a webpage:
df_whole_csp_set=pd.concat(whole_csp_set,ignore_index=True,axis=0)
df_whole_csp_set.to_csv(file_path+"/df_whole_set.csv")
df_whole_csp_set=df_whole_csp_set[df_whole_csp_set["ctype"]=="CSP"]
df_whole_csp_set.to_csv(file_path+"/df_whole_csp_set.csv")
enforcement_homepage_mulcsp =[]
enforcement_homepage_mulcsp_diff=[]
print("number of home page use csp:",len(df_whole_csp_set))
for index,row in df_whole_csp_set.iterrows():
    if(row["cspmetaheader"]!="none_csp" and row["cspheader"]!="designed_CSP_only"):
        enforcement_homepage_mulcsp.append(row["domain"])
        if(row["cspmetacon"]!=row["cspcontent"]):
            enforcement_homepage_mulcsp_diff.append(row["domain"])
            print("##################################   ",row["domain"])
            print("mode:",row["cspmetaheader"],"cspmeta:", row["cspmetacon"])
            print("mode",row["cspheader"],"cspheader:", row["cspcontent"])
print("mul dom:",enforcement_homepage_mulcsp)
print("number of mul_dom:",len(list(set(enforcement_homepage_mulcsp))))
print("number of mul_dom_diff:",len(list(set(enforcement_homepage_mulcsp_diff))))
#check whether there are some domain deploy multiple csp by multiple csp header
d_group=df_whole_csp_set.groupby(df_whole_csp_set["domain"],as_index=False)
for g_iname, g in d_group:
    if(len(g)>1):
        enforcement_homepage_mulcsp_diff.append(g["domain"].iloc[0])
print("number of mul_dom_diff:",len(list(set(enforcement_homepage_mulcsp_diff))))


##################extract csp features


