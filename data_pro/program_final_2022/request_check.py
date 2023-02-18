import json
import difflib
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

file_path="/Volumes/Elements/CSPNEW_DATA/"
#Datafolder=["CSPNEW02"]
Datafolder=["CSPNEW01","CSPNEW02","CSPNEW03","CSPNEW04","CSPNEW05-1","CSPNEW05-2","CSPNEW06","CSPNEW07-1","CSPNEW07-2","CSPNEW08-1","CSPNEW08-2","CSPNEW09-1","CSPNEW09-2","CSPNEW10-1","CSPNEW10-2"]
i=0
for dafolder in Datafolder:
    file_name=file_path+dafolder
    file_name_con = file_name+"/df_concat_2.csv"
    csv_data = pd.read_csv(file_name_con, low_memory=False)
    df_con = pd.DataFrame(csv_data)
    df_con=df_con[['site_url_y','dom',"domain"]]
    df_con.drop_duplicates(["site_url_y", "dom"], 'last', inplace=True)

    file_name_con2 = "/Volumes/Elements/CSP_HOMEPAGE/df_merge_csp.csv"
    csv_data2 = pd.read_csv(file_name_con2, low_memory=False)
    df_con2 = pd.DataFrame(csv_data2)
    domains=df_con2[(df_con2["domain"]-10000*(i+1)<10000) & (df_con2["domain"]-10000*(i+1)>0)]["domain"]-10000*(i+1)
    print("domain len:",len(domains))

    # df_con=df_con[df_con["domain"].isin(domains)]
    # df_con.drop_duplicates(["site_url_y","dom"], 'last', inplace=True)
    # df_con.to_csv(file_path + "/CSP_CLUSTER/df_csp_html"+str(i)+".csv")
    # print("length of site set",len(list(set(df_con["site_url_y"]))))
    # print("length of original sites:", len(list(set(df_con["site_url_y"]))))

    ######################################################################################
    file_name = file_path + dafolder
    file_name_con = file_name + "/df_concat_req.csv"
    csv_data = pd.read_csv(file_name_con, low_memory=False)
    df_req = pd.DataFrame(csv_data).drop(columns=["dom"])
    df_req=df_req[df_req["domain"].isin(domains)]
    df_req.to_csv(file_path + "/CSP_CLUSTER/df_csp_req" + str(i) + ".csv")
    ########################################################################################
    # file_name = file_path + dafolder
    # file_name_con = file_name + "/df_concat_log.csv"
    # csv_data = pd.read_csv(file_name_con, low_memory=False)
    # df_log = pd.DataFrame(csv_data).drop(columns=["dom"])
    # df_log = df_log[df_log["site_url_y"].isin(site_list)]
    # df_log.to_csv(file_path + "/CSP_CLUSTER/df_csp_log" + str(i) + ".csv")
    i=i+1
