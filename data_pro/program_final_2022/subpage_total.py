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
# all_len=0
# all_len2=0
# all_len3=0
# for fold_num in range(1,16):
#     file_name_con = "/Volumes/Elements/CSP_HOMEPAGE/df_concat_encsp"+str(fold_num)+".csv"
#     csv_data = pd.read_csv(file_name_con, low_memory=False)
#     df_csp_subpage = pd.DataFrame(csv_data)
#     df_set=df_csp_subpage.drop_duplicates(["domain","site_url_y"])
#     all_len=all_len+len(df_csp_subpage)
#     all_len2=all_len2+len(df_set)
#     all_len3=all_len3+len(df_set[(df_set["type"]=="sub_domain") & (df_set["ctype"]=="NCSP")])
# print(all_len)
# print(all_len2)
# print(all_len3)
#
# file_name_con = "/Volumes/Elements/CSPNEW_DATA/df_csp_normfeature0.csv"
# csv_data = pd.read_csv(file_name_con, low_memory=False)
# df_encsp_all = pd.DataFrame(csv_data)
# print(len(df_encsp_all[df_encsp_all["ctype"]=="NCSP"]))
#
# index_set=[]
# for ind, row in df_encsp_all.iterrows():
#     sum_col=0
#     for col in df_encsp_all.columns:
#         if(col!="Site" and col!="csp_con" and col!="domain"	and col!="csp_mode"	and col!="ctype" and col!="type" and col!="Unnamed: 0"):
#             sum_col=sum_col+row[col]
#     if(sum_col==0):
#         index_set.append(ind)
# print("ncsp:", len(df_encsp_all[(df_encsp_all.index.isin(index_set)) & (df_encsp_all["ctype"]=="NCSP")]))
# print("csp:", len(df_encsp_all[(df_encsp_all.index.isin(index_set)) & (df_encsp_all["ctype"]=="CSP")]))
#
# #merge csp
# print(len(df_encsp_all))
# csp_group=df_encsp_all.groupby(["domain","Site","csp_mode"])
# print("groupn len:",len(csp_group))
# for gn,g in csp_group:
#           if(len(g)>1):
#               d = g.iloc[0]
#               inde=g.index[0]
#               print("inde",inde)
#               for col in df_encsp_all.columns:
#                 #if (col != "Site" and col != "csp_con" and col != "domain" and col != "csp_mode"):
#                 if (col != "Site" and col != "csp_con" and col != "domain" and col != "csp_mode" and col != "ctype" and col != "type" and col != "Unnamed: 0"):
#                   ini_val=d[col]
#                   for index, de in g.iterrows():
#                       ini_val=ini_val+de[col]
#                   if(ini_val>1):
#                      ini_val=1
#                   df_encsp_all.at[inde,col]=ini_val      #print(d[col])
#               for index, de in g.iterrows():
#                   if(index!=inde):
#                     df_encsp_all.drop(index=index,inplace=True)
# print(len(df_encsp_all))
# df_encsp_all.to_csv("/Volumes/Elements/CSPNEW_DATA/df_merge_csp.csv")

#####
csv_data = pd.read_csv("/Volumes/Elements/CSPNEW_DATA/df_merge_csp.csv", low_memory=False)
df_csp_subpage = pd.DataFrame(csv_data)
s_group = df_csp_subpage.groupby(["domain"])
no_subpage=[]
csp_site_with_sp=[]
for g_n, g in s_group:
    if(len(list(set(g["Site"])))==1):
        no_subpage.append(g.iloc[0]["domain"])
        #print(g["type"])
    else:
        csp_site_with_sp.append(g.iloc[0]["domain"])
print("website without a collected subpage:",len(list(set(no_subpage))))
csp_site_with_sp=df_csp_subpage[df_csp_subpage["domain"].isin(csp_site_with_sp)]

csp_site_with_sg=csp_site_with_sp.groupby("domain")
print(len(csp_site_with_sg))
all_sub_csp=[]
part_sub_csp=[]
all_sub_no_csp=[]
for gn, g in csp_site_with_sg:
    domain=g.iloc[0]["domain"]
    ctype=[]
    for ind, csp in g.iterrows():
        if(csp["type"]=="sub_domain"):
           ctype.append(csp["ctype"])
    ctype=list(set(ctype))
    if(("CSP" in ctype) and ("NCSP" not in ctype)):
        all_sub_csp.append(domain)
    if(("CSP" in ctype) and ("NCSP" in ctype)):
        part_sub_csp.append(domain)
    if(("CSP" not in ctype) and ("NCSP" in ctype)):
        all_sub_no_csp.append(domain)
print(len(all_sub_csp))
print(len(part_sub_csp))
print(len(all_sub_no_csp))

csp_site_allcsp=csp_site_with_sp[csp_site_with_sp["domain"].isin(all_sub_csp)]
csp_site_allcsp_g=csp_site_allcsp.groupby("domain")
all_csp_en=[]
all_csp_bothm=[]
for gn, g in csp_site_allcsp_g:
    domain=g.iloc[0]["domain"]
    cmode=[]
    for ind, csp in g.iterrows():
        if(csp["type"]=="sub_domain"):
           cmode.append(csp["csp_mode"])
    cmode = list(set(g["csp_mode"]))
    if (("content-security-policy" in cmode) and ("content-security-policy-report-only" not in cmode)):
        all_csp_en.append(domain)
    if (("content-security-policy" in cmode) and ("content-security-policy-report-only" in cmode)):
        all_csp_bothm.append(domain)
print("second stage:",len(all_csp_en))
print("second stage:",len(all_csp_bothm))

all_csp_all_en=csp_site_allcsp[csp_site_allcsp["domain"].isin(all_csp_en)]
all_csp_all_en_g=all_csp_all_en.groupby("domain")
all_csp_same_tem=[]
all_same_con=[]
for gn, g in all_csp_all_en_g:
    domain=g.iloc[0]["domain"]
    for ind, en in g.iterrows():
        if(en["type"]=="domain"):
            or_csp=en
        else:
            not_same=0
            not_same_con=0
            if(en["csp_con"]!=or_csp["csp_con"]):
                not_same_con=1
            for col in all_csp_all_en.columns:
                if(col!="Site" and col!="csp_mode" and col!="csp_con" and col!="type" and col!="ctype" and col!="domain"):
                    if(en[col]!=or_csp[col]):
                        not_same=1
                        #print(col)
    if(not_same==0):
       all_csp_same_tem.append(domain)
    if(not_same_con==0):
       all_same_con.append(domain)
print("second stage:",len(all_csp_same_tem))
print("second stage:",len(all_same_con))



all_csp_all_both=csp_site_allcsp[csp_site_allcsp["domain"].isin(all_csp_bothm)]
print(set(all_csp_all_both["csp_mode"]))
all_csp_all_both=all_csp_all_both.drop(all_csp_all_both[all_csp_all_both["csp_mode"]=="content-security-policy-report-only"].index)
print(set(all_csp_all_both["csp_mode"]))
all_csp_all_both_g=all_csp_all_both.groupby("domain")
all_csp_all_en=csp_site_allcsp[csp_site_allcsp["domain"].isin(all_csp_en)]
all_csp_all_en_g=all_csp_all_en.groupby("domain")
all_csp_same_tem_bo=[]
all_same_con_bo=[]
for gn, g in all_csp_all_both_g:
  domain=g.iloc[0]["domain"]
  if(len(g)>1):
    for ind, en in g.iterrows():
        if(en["type"]=="domain"):
            or_csp=en
        else:
            not_same=0
            not_same_con=0
            if(en["csp_con"]!=or_csp["csp_con"]):
                not_same_con=1
            for col in all_csp_all_en.columns:
                if(col!="Site" and col!="csp_mode" and col!="csp_con" and col!="type" and col!="ctype" and col!="domain"):
                    if(en[col]!=or_csp[col]):
                        not_same=1
                        #print(col)
    if(not_same==0):
       all_csp_same_tem_bo.append(domain)
    if(not_same_con==0):
       all_same_con_bo.append(domain)
print("third stage:",len(all_csp_same_tem_bo))
print("third stage:",len(all_same_con_bo))






print("################################################################################")
csp_site_partcsp=csp_site_with_sp[csp_site_with_sp["domain"].isin(part_sub_csp)]
print(set(csp_site_partcsp["ctype"]))
print(len(list(set(csp_site_partcsp["domain"]))))
csp_site_partcsp=csp_site_partcsp.drop(csp_site_partcsp[csp_site_partcsp["ctype"]=="NCSP"].index)
print(set(csp_site_partcsp["ctype"]))



csp_site_partcsp_gg=csp_site_partcsp.groupby("domain")
all_csp_en_411=[]
all_csp_bothm_411=[]
for gn, g in csp_site_partcsp_gg:
    domain=g.iloc[0]["domain"]
    cmode = []
    for ind, csp in g.iterrows():
        if (csp["type"] == "sub_domain"):
            cmode.append(csp["csp_mode"])
    cmode = list(set(g["csp_mode"]))
    if (("content-security-policy" in cmode) and ("content-security-policy-report-only" not in cmode)):
        all_csp_en_411.append(domain)
    if (("content-security-policy" in cmode) and ("content-security-policy-report-only" in cmode)):
        all_csp_bothm_411.append(domain)
print("forth stage:",len(all_csp_en_411))
print("forth stage:",len(all_csp_bothm_411))

csp_site_partcsp=csp_site_partcsp.drop(csp_site_partcsp[csp_site_partcsp["csp_mode"]=="content-security-policy-report-only"].index)
print(set(csp_site_partcsp["csp_mode"]))
csp_site_partcsp_g=csp_site_partcsp.groupby("domain")
all_csp_same_tem_pa=[]
all_same_con_pa=[]
for gn, g in csp_site_partcsp_g:
   domain=g.iloc[0]["domain"]
   if(len(g)<2):
      print("domain report all:",domain)
   if(len(g)>1):
    for ind, en in g.iterrows():
        if(en["type"]=="domain"):
            or_csp=en
        else:
            not_same=0
            not_same_con=0
            if(en["csp_con"]!=or_csp["csp_con"]):
                not_same_con=1
            for col in all_csp_all_en.columns:
                if(col!="Site" and col!="csp_mode" and col!="csp_con" and col!="type" and col!="ctype" and col!="domain"):
                    if(en[col]!=or_csp[col]):
                        not_same=1
                        #print(col)
    if(not_same==0):
       all_csp_same_tem_pa.append(domain)
    if(not_same_con==0):
       all_same_con_pa.append(domain)
print("third stage:",len(all_csp_same_tem_pa))
print("third stage:",len(all_same_con_pa))


