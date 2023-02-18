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

resource_type_all=["main_frame", "sub_frame", "stylesheet", "script", "image", "font", "object", "xmlhttprequest", "ping", "csp_report", "media", "websocket", "other"]
csp_concat_html_set=[]
csp_concat_req_set=[]
csp_concat_log_set=[]

#Datafolder=["CSPNEW01","CSPNEW02","CSPNEW03","CSPNEW04","CSPNEW05-1","CSPNEW05-2","CSPNEW06","CSPNEW07-1","CSPNEW07-2","CSPNEW08-1","CSPNEW08-2","CSPNEW09-1","CSPNEW10-1","CSPNEW09-2","CSPNEW10-2"]

file_path="/Volumes/Elements/CSPNEW_DATA/CSP_CLUSTER/"
Datafolder= 15
csp_concat_vec_set=[]
for i in range(0,Datafolder):
    file_name_con = file_path+"df_csp_html"+str(i)+".csv"
    csv_data = pd.read_csv(file_name_con, low_memory=False)
    df_csp_html = pd.DataFrame(csv_data)
    df_csp_html.drop_duplicates(["site_url_y"], 'last', inplace=True)
    csp_concat_html_set.append(df_csp_html)
    print("df_csp_html:",len(df_csp_html))

    file_name_con = file_path+"df_csp_req"+str(i)+".csv"
    csv_data = pd.read_csv(file_name_con, low_memory=False)
    df_csp_req = pd.DataFrame(csv_data)
    csp_concat_req_set.append(df_csp_req)
    print("df_csp_req:", len(list(set(df_csp_req["site_url_y"]))))
#######concat CSP vec
df_csp_html_all = pd.concat(csp_concat_html_set,ignore_index=True,axis=0)
df_csp_html_all.to_csv((file_path+"/df_all_html.csv"))
df_csp_html_all=df_csp_html_all.drop(columns=['Unnamed: 0'])

df_csp_req_all = pd.concat(csp_concat_req_set,ignore_index=True,axis=0)
df_csp_req_all.to_csv((file_path+"/df_all_req.csv"))
df_csp_req_all=df_csp_req_all.drop(columns=['Unnamed: 0'])

#######################################################################eval check
# df_eval_check= pd.DataFrame(columns=["Site",])
# for index, row in df_csp_html_all.iterrows():
#     Site=row["site_url_y"]
#     eval=0
#     if(row["dom"].find("eval(")>0):
#         eval=1
#     df_eval_check.loc[index] = [Site, eval]
#######################################################################prefetch/manifest
df_dom_check= pd.DataFrame(columns=["Site","resource_type_set"])
ind=0
for index, row in df_csp_html_all.iterrows():
    res_set=""
    resource_set=[]
    #row["dom"]="<form action=\"javascript:alert('Foo')\" id=\"form1\" method=\"post\">"
    soup = BeautifulSoup(row["dom"], 'html.parser')
    if(len(soup.find_all(attrs={'rel': 'manifest'}))>0):
        resource_set.append("manifest")
    if(len(soup.find_all("form",attrs={'action': True}))>0):
        resource_set.append("form-action")
    if(len(soup.find_all(attrs={'rel': 'prefetch'}))>0 or len(soup.find_all(attrs={'rel': 'prerender'}))>0):
        resource_set.append("prefetch")
    if(len(soup.find_all("base"))>0):
        resource_set.append("base")
    # if(len(soup.find_all(attrs={'src': True}))>0):
    #   src_set=soup.find_all(attrs={'src': True})
    #   for src in src_set:
    #     if src.find("http:"):
    #      resource_set.append("http")
    # if(len(soup.find_all(attrs={'src': True}))>0):
    #   src_set=soup.find_all(attrs={'src': True})
    #   for src in src_set:
    #     if src.find("http:"):
    #      resource_set.append("http")
    res_set=" ".join(resource_set)
    df_dom_check.loc[ind] =[row['site_url_y'], res_set]
    ind=ind+1
df_dom_check.to_csv((file_path+"/df_dom_check.csv"))
############################################################################req analysis
df_req_check= pd.DataFrame(columns=["Site","req_type_set"])
req_group=df_csp_req_all.groupby(df_csp_req_all['site_url_y'],as_index=False)
ind=0
for rgroup in req_group:
    reqt_set=''
    req_type_set=[]
    req_type_set=list(set(rgroup[1]['contentType']))
    if ("csp_report" in req_type_set):
       req_type_set=["" if val=="csp_report" else val for val in req_type_set]
    if ("other" in req_type_set):
        req_type_set = ["" if val == "other" else val for val in req_type_set]
    if ("main_frame" in req_type_set):
       req_type_set=["" if val=="main_frame" else val for val in req_type_set]
    if("ping" in req_type_set):
        req_type_set=["connect" if val=="ping" else val for val in req_type_set]
    if("xmlhttprequest" in req_type_set):
        req_type_set = ["connect" if val == "xmlhttprequest" else val for val in req_type_set]
    if("websocket" in req_type_set):
        req_type_set = ["connect" if val == "websocket" else val for val in req_type_set]
    if("sub_frame" in req_type_set):
        req_type_set = ["frame" if val == "sub_frame" else val for val in req_type_set]
    if("stylesheet" in req_type_set):
        req_type_set = ["style" if val == "stylesheet" else val for val in req_type_set]
    req_type_set=list(set(req_type_set))
    reqt_set=" ".join(req_type_set)
    df_req_check.loc[ind]=[rgroup[1]['site_url_y'].iloc[0],reqt_set]
    ind=ind+1
df_req_check.to_csv((file_path+"/df_req_check.csv"))
############################################################################log_analysis
#log_group=df_csp_log_all.groupby(df_csp_log_all['DocumentUri'],as_index=False)
dfs = [df_dom_check, df_req_check]
df_all_check = reduce(lambda left, right: pd.merge(left, right,how="left", on='Site'), dfs)
df_all_check.to_csv((file_path+"/df_all_check.csv"))
df_all_check.fillna("", inplace=True)
for index,row in df_all_check.iterrows():
    Site=row["Site"]
    #print(row["req_type_set"]== null)
    #print(row["resource_type_set"]=="")
    if(row["resource_type_set"]!=""):
       df_all_check.at[index,"req_type_set"]=row["req_type_set"]+" "+row["resource_type_set"]
df_all_check.drop(columns=["resource_type_set"],inplace=True)
df_all_check.to_csv((file_path+"/df_all_check_2.csv"))

###################################################################################################3
file_name= file_path + "df_all_features.csv"
csv_data = pd.read_csv(file_name, low_memory=False)
df_csp_all = pd.DataFrame(csv_data)
df_csp_all["coverage"]=0
df_csp_all["rs2"]=""
df_csp_all["rs"]=""
for index,row in df_csp_all.iterrows():
    Site=row["Site"]
    res_set=[]
    for col in df_csp_all.columns:
        if(col!="lab12" and col!="Unnamed: 0" and col!="Site" and col!="coverage" and col!="rs2" and col!="rs"):
            cval=col.split("_")[0]
            cval=cval.split("-src")[0]
            cval = cval.split("-uri")[0]
            #if(cval=="block-all-mixed-content" and row[col]==1):
               #cval="frame"
                #res_set.append(cval)
            if(cval=="default" and row[col]==1):
                res_set.append("script")
                res_set.append("frame")
                res_set.append("image")
                res_set.append("object")
                res_set.append("style")
                res_set.append("media")
                res_set.append("connect")
                res_set.append("manifest")
                res_set.append("prefetch")
                res_set.append("font")
            elif(cval=="child" and row[col]==1):
                res_set.append("frame")
                res_set.append("script")
            # if(cval=="frame-ancestor"):
            #     cval="frame"
            elif(cval=="img" and row[col]==1):
                cval="image"
                res_set.append(cval)
            # elif (cval == "plugin-types" and row[col]==1):
            #     cval = "object"
            #     res_set.append(cval)
            elif (cval == "require-sri-for" and row[col]==1):
                if(col=="require-sri-for_script"):
                   cval = "script"
                   res_set.append(cval)
                if(col=="require-sri-for_style"):
                   cval = "style"
                   res_set.append(cval)
            elif (cval == "require-trusted-types-for" and row[col] == 1):
                cval = "script"
                res_set.append(cval)
            elif (cval == "trusted-types" and row[col] == 1):
                cval = "script"
                res_set.append(cval)
            # if (cval == "upgrade-insecure-requests" and row[col] == 1):
            #     cval = "http"
            elif (cval == "worker" and row[col] == 1):
                cval = "script"
                res_set.append(cval)
            elif(row[col] == 1):
                res_set.append(cval)
    rs=set(res_set)
    df_csp_all.at[index, "rs"] = rs
    rs2=df_all_check[df_all_check["Site"]==Site].iloc[0]["req_type_set"].split(" ")
    rs2=filter(None,rs2)
    rs2=set(rs2)
    if(len(rs2)==0):
        df_csp_all.at[index, "coverage"] = 1
    else:
     df_csp_all.at[index, "rs2"] = rs2
     cm=rs.intersection(rs2)
     print("res2:",rs2)
     print("res1:",rs)
     if (rs2==cm):
            df_csp_all.at[index, "coverage"] = 1
df_csp_all.to_csv((file_path+"/df_all_csp_coverage.csv"))


