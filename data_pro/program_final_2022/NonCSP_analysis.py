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

# tldextract.extract(("http://www.badu.com")).
file_path="/Volumes/Elements/CSPNEW_DATA/"
#Datafolder=["CSPNEW02"]
Datafolder=["CSPNEW01","CSPNEW02","CSPNEW03","CSPNEW04","CSPNEW05-1","CSPNEW05-2","CSPNEW06","CSPNEW07-1","CSPNEW07-2","CSPNEW08-1","CSPNEW08-2","CSPNEW09-1","CSPNEW09-2","CSPNEW10-1","CSPNEW10-2"]
#dropped_csp=[]
#####step 1:filter NCSP domains
NCSP_domain_list=[]
len_all=0
# for dafolder in Datafolder:
#     file_name=file_path+dafolder
#     file_name_con = file_name+"/df_concat_sub_3.csv"
#     csv_data = pd.read_csv(file_name_con, low_memory=False)
#     df_con = pd.DataFrame(csv_data)
#     df_con=df_con[(df_con["ctype"]=="NCSP") & (df_con["type"]=="domain")]
#     #df_con["domain"]=df_con["domain"]+10000*fold_num
#     NCSP_domain_list.append(list(set(df_con["domain"])))
#     # len_all=len_all+len(df_con.groupby("domain"))
# #     print("domain len:",len(list(set(df_con["domain"]))))
# #     print("************************************************************")
# # print("len of df_con:",len_all)
#
# #####step2:filter ncsp
# ncsp_concat_vec_set=[]
# fold_num=0
# for dafolder in Datafolder:
#     fold_num=fold_num+1
#     file_name=file_path+dafolder
#     file_name_con = file_name+"/df_concat_sub_3.csv"
#     csv_data = pd.read_csv(file_name_con, low_memory=False)
#     df_con = pd.DataFrame(csv_data)
#     domain=NCSP_domain_list[fold_num-1]
#     # print("domain len:",len(domain))
#     df_con=df_con[df_con["domain"].isin(domain)]
#     df_con["domain"]=df_con["domain"]+10000*fold_num
#     ncsp_concat_vec_set.append(df_con)
# #concat all NCSP websites
# df_ncsp_all = pd.concat(ncsp_concat_vec_set,ignore_index=True,axis=0)
# df_ncsp_all.to_csv((file_path+"/df_all_ncsp.csv"))
# df_ncsp_all=df_ncsp_all.drop(columns=['Unnamed: 0'])
# print("number of NCSP websites:",len(df_ncsp_all.groupby("domain")))
#
# ######step3: filter websites: homepage is NCSP, but some subpages are CSPs
# all_group = df_ncsp_all.groupby("domain")
# ncsp_website_only_homepage=[]
# all_ncsp_website=[]
# subpage_csp_website=[]
# for gname, allg in all_group:
#         #print(allg["ctype"])
#         dmain=""
#         doma=-1
#         n_ncsp=0
#         n_csp=0
#         do_csp=0
#         enforce=0
#         if (len(allg) == 1 and allg.iloc[0]["type"] == "domain"):
#             ncsp_website_only_homepage.append(allg.iloc[0]["domain"])
#         else:
#          for index, al in allg.iterrows():
#             if(al["type"] == "domain"):
#                dmain=al["site_url_y"]
#                doma=al["domain"]
#                if(al["ctype"]=="CSP"):
#                    do_csp=1
#             if(al["ctype"]=="NCSP"):
#                 n_ncsp=n_ncsp+1
#             if(al["ctype"]=="CSP"):
#                 n_csp=n_csp+1
#         if(do_csp==0 and n_ncsp>0 and n_csp==0):
#             all_ncsp_website.append(doma)
#         if ( n_csp>0 and do_csp==0):
#             subpage_csp_website.append(doma)
# print("ncsp_website_only_homepage:", len(list(set(ncsp_website_only_homepage))))
# print("all_ncsp_website:",len(all_ncsp_website))
# print("subpage_csp_website:", len(subpage_csp_website))
# #####################################################step4:concat with violations
# fold_num=0
# ncsp_concat_log_set=[]
# for dafolder in Datafolder:
#     fold_num = fold_num + 1
#     file_name=file_path+dafolder
#     file_name_con = file_name + "/df_logs_1.csv"
#     csv_data = pd.read_csv(file_name_con, low_memory=False)
#     df_logs_1 = pd.DataFrame(csv_data)
#     # remove duplicates
#     group_logs = df_logs_1.groupby('DocumentUri')
#     start_time = group_logs["Site_num"].min();
#     gro_num = 0;
#     gro_len = 0;
#     print("group:", len(group_logs))
#     for name, group in group_logs:
#         for index, row in group.iterrows():
#             if (float(row["Site_num"]) - float(start_time[gro_num]) > 60000):
#                 # print("del")
#                 group.drop(index=index, inplace=True)
#         gro_num = gro_num + 1
#         gro_len = gro_len + len(group)
#     domain_1 = NCSP_domain_list[fold_num-1]
#     domain=[i + fold_num*10000 for i in domain_1]
#     df_ncsp_sub=df_ncsp_all[df_ncsp_all["domain"].isin(domain)]
#     dfs_3 = [df_ncsp_sub, df_logs_1]
#     df_concat_log = reduce(lambda left, right: pd.merge(left, right, on='DocumentUri'), dfs_3)
#     ncsp_concat_log_set.append(df_concat_log)
#
# df_ncsp_all = pd.concat(ncsp_concat_log_set,ignore_index=True,axis=0)
# df_ncsp_all.to_csv((file_path+"/df_all_log_ncsp.csv"))
#
#
# # dfs_4 = [df_ncsp_all, df_req_1]
# # df_req_1.rename(columns={'site': 'site_url_y'}, inplace=True)
# # df_concat_req = reduce(lambda left, right: pd.merge(left, right, on='site_url_y'), dfs_4)
# # df_concat_req.to_csv((file_name + "/df_concat_req.csv"))
file_name_con = file_path+"/df_all_log_ncsp.csv"
csv_data = pd.read_csv(file_name_con, low_memory=False)
df_ncsp_all = pd.DataFrame(csv_data)
print(len(df_ncsp_all))
#########################################################step5:NCSP homepage analysis
df_ncsp_all_homepage=df_ncsp_all[(df_ncsp_all["type"]=="domain") & (df_ncsp_all["ctype"]=="NCSP")]
print("homepages with CSP violations:",len(list(set(df_ncsp_all_homepage["domain"]))))
#########type of CSP violation distribution
for col in df_ncsp_all_homepage.columns:
    print("col:",col)
total_nonp=df_ncsp_all_homepage.groupby(df_ncsp_all_homepage['DocumentUri'],as_index=False)
print("number of noncsp webpages with violations:",len(total_nonp))
#ncsp_logs.to_csv('/Users/mengxiaren/Desktop/CSPDB/CSP_Data/nonlogs.csv')
#ncsp_logs=df_logs[(df_logs['DocumentUri'].isin(ncsp_site)) | (df_logs['Referrer'].isin(ncsp_site))]
#print("total number of logs of none_csp websites",(len(ncsp_logs)))
################################################################################violations for each page
vio_num=[]
for df_sub in total_nonp:
  total_count=len(df_sub)
  vio_num.append(total_count)
  # if(total_count>3000):
  #     print("violation is 3000:",df_sub[1]['siteurl'].iloc[0])
  # if (total_count ==1):
  #     print("violation is 1:",df_sub[1]['siteurl'].iloc[0])
vio_num=np.array(vio_num)
for i in vio_num:
 print("vio_num:",i)
print("average number of violations for valid ncsp homepages:",np.mean(vio_num)," with max_num:",np.max(vio_num),"and min_num:",np.min(vio_num))

# vio_wnum=[]
# for df_sub in total_nonw:
#   total_count=len(df_sub)
#   vio_wnum.append(total_count)
# vio_wnum=np.array(vio_wnum)
# print("average number of violations for valid ncsp websites:",np.mean(vio_wnum)," with max_num:",np.max(vio_wnum),"and min_num:",np.min(vio_wnum))
#####is the violation only for fetch directives?
violation_type_2=df_ncsp_all_homepage['EffectiveDirective'].unique()
print("number of violations type:",len(violation_type_2))
violation_distri_2={}
#violation_distri_site_2={}
for vio in violation_type_2:
    vc=len(df_ncsp_all_homepage[(df_ncsp_all_homepage['EffectiveDirective']== vio)])
    violation_distri_2.update({vio : vc})
print(violation_distri_2)

#########################################################[eval,inline] violations in none csp website[general]
violation_distri_3={}
#violation_distri_site_3={}
violation_type_3=['inline','eval']
for vio in violation_type_3:
    vc=len(df_ncsp_all_homepage[(df_ncsp_all_homepage['BlockedUri']== vio)])
    violation_distri_3.update({vio : vc})
print(violation_distri_3)

#########################################################violations of websites
violation_distri_22={}
violation_distri_site_22={}
vio_distri_site_all=[]
vn=0
g_nonp_vio=df_ncsp_all_homepage.groupby(df_ncsp_all_homepage['domain'],as_index=False)
for subg in g_nonp_vio:
    violation_distri_22={'child-src':0,'img-src': 0, 'font-src': 0, 'style-src-elem': 0, 'script-src-elem': 0, 'script-src': 0, 'script-src-attr': 0, 'style-src-attr': 0, 'frame-src': 0, 'connect-src': 0, 'media-src': 0, 'worker-src': 0, 'object-src': 0, 'prefetch-src': 0, 'manifest-src': 0}
    for eff_dire in (subg[1]['EffectiveDirective']):
            vn= violation_distri_22[eff_dire]+1
            violation_distri_22.update({eff_dire: vn})
    vio_distri_site_all.append(violation_distri_22)
#print(vio_distri_site_all) ###violations of different dir for each pages

#########################################################types of violations in none csp website
violation_distri_4={}
violation_distri_site_4={}
violation_type_4=['inline','eval']
script_dir=["script-src","script-src-attr","script-src-elem"]
style_dir=["style-src","style-src-attr","style-src-elem"]
for vio in violation_type_4:
    vc1=len(df_ncsp_all_homepage[(df_ncsp_all_homepage['BlockedUri']== vio) & (df_ncsp_all_homepage['EffectiveDirective'].isin(style_dir))])
    name_style="style_"+vio
    violation_distri_4.update({name_style : vc1})

    vc2 = len(df_ncsp_all_homepage[(df_ncsp_all_homepage['BlockedUri'] == vio) & (df_ncsp_all_homepage['EffectiveDirective'].isin(script_dir))])
    name_script = "script_" + vio
    violation_distri_4.update({name_script: vc2})
print(violation_distri_4)

###################################################################details distribution of eval and inline
st_dir=["style-src","style-src-attr","style-src-elem","script-src","script-src-attr","script-src-elem"]
####for inline
violation_distri_5={}
for dir in st_dir:
    vcs = len(df_ncsp_all_homepage[(df_ncsp_all_homepage['BlockedUri'] == 'inline') & (df_ncsp_all_homepage['EffectiveDirective']==dir)])
    name_script = "inline_" + dir
    violation_distri_5.update({name_script: vcs})
print(violation_distri_5)

####for eval
violation_distri_6={}
for dir in st_dir:
    vcs = len(df_ncsp_all_homepage[(df_ncsp_all_homepage['BlockedUri'] == 'eval') & (df_ncsp_all_homepage['EffectiveDirective']==dir)])
    name_script = "eval_" + dir
    violation_distri_6.update({name_script: vcs})
print(violation_distri_6)

###################################################################simplification
##self per page
self_num=0
self_page_num=0
th_self=5
self_distri_all = []

for df_sub in total_nonp:
  total_count=len(df_sub[1])
  docuri = (df_sub[1]['DocumentUri']).iloc[0]
  docuri_1 = docuri.split("/")[0]
  docuri_2 = docuri.split("/")[2]
  self_distri = {'docuri': docuri,'default-src':0,'style-src':0,'child-src':0,'img-src': 0, 'font-src': 0, 'style-src-elem': 0, 'script-src-elem': 0, 'script-src': 0, 'script-src-attr': 0, 'style-src-attr': 0, 'frame-src': 0, 'connect-src': 0, 'media-src': 0, 'worker-src': 0, 'object-src': 0, 'prefetch-src': 0, 'manifest-src': 0}
  for blocku,viodir in zip(df_sub[1]['BlockedUri'],df_sub[1]['EffectiveDirective']):
     blocku_or_1 = blocku.split("/")[0]
     if (blocku_or_1=="https:" or blocku_or_1=="http:"):
         blocku_or_2 = blocku.split("/")[2]
         if (blocku_or_1==docuri_1 and blocku_or_2==docuri_2):
             vc = self_distri[viodir]+1
             self_distri.update({viodir: vc})
  self_distri_all.append(self_distri)
#print("number of pages containing 'self' elements:",self_distri_all)
#print("number of pages containing 'self' elements:",len(self_distri_all))
###

#######################################self statistic for each directive
self_for_dir = {'default-src':0,"style-src":0,'child-src':0,'img-src': 0, 'font-src': 0, 'style-src-elem': 0, 'script-src-elem': 0, 'script-src': 0, 'script-src-attr': 0, 'style-src-attr': 0, 'frame-src': 0, 'connect-src': 0, 'media-src': 0, 'worker-src': 0, 'object-src': 0, 'prefetch-src': 0, 'manifest-src': 0}
for sl in self_distri_all:
    for key in sl.keys():
        if(key!='docuri' and sl[key]>0):
            self_for_dir[key]=self_for_dir[key]+1;
print(" 'self' elements statistic for each dir(how many pages need self for a dir):", self_for_dir)
########################################
#######################################self directives statistic for each page
self_dir_page=[]
for sl in self_distri_all:
    self_dir=0;
    for key in sl.keys():
        if(key!='docuri' and sl[key]>0):
            self_dir=self_dir+1;
    self_dir_page.append(self_dir)
print(" 'self' elements statistic for each dir:", self_dir_page)
self_dir_page_dic=Counter(self_dir_page)
print(" 'self' elements statistic for each dir:",self_dir_page_dic)
########################################

#######################################self statistic for each page
self_for_page_ls=[]
self_for_page_dict=[]
for sl in self_distri_all:
   self_for_page=0
   for key in sl.keys():
       if(key!='docuri'):
            self_for_page=self_for_page+sl[key];
   self_for_page_ls.append(self_for_page)
print(" 'self' elements statistic for each page:", self_for_page_ls)
self_for_page_dict=Counter(self_for_page_ls)
print(" 'self' elements statistic for each page:", self_for_page_dict)
#print(" 'self' elements for pages maximum:", max(self_for_page_ls))
#print(" 'self' elements for pages minimum:", min(self_for_page_ls))
#print(" 'self' elements for pages average:", np.mean(self_for_page_ls))
########################################

##default self for  per page
# default_self=0;
# for self in self_distri_all:
#   snum = 0;
#   for value in self.values():
#        if type(value) == type(3):
#           if value > 0:
#             snum=snum+1
#   if snum==14:
#        default_self=default_self+1
# print("number of pages can use default-src self:", default_self)

self_sta = {'img-src': 0, 'font-src': 0, 'style-src-elem': 0, 'script-src-elem': 0, 'script-src': 0, 'script-src-attr': 0, 'style-src-attr': 0, 'frame-src': 0, 'connect-src': 0, 'media-src': 0, 'worker-src': 0, 'object-src': 0, 'prefetch-src': 0, 'manifest-src': 0}
for i in range(len(self_distri_all)):
    for key in self_sta.keys():
        self_sta[key]=self_sta[key]+self_distri_all[i][key]
print("self for url:",self_sta)
#####################################################################
#########################################################non-removable
img_url=[]
font_url=[]
style_elem_url=[]
script_elem_url=[]
connect_url=[]
frame_url=[]
nonp_unique=df_ncsp_all.groupby(["domain",'DocumentUri','EffectiveDirective'],as_index=False)
for sub_vio in nonp_unique:
    docuri = (sub_vio[1]['DocumentUri']).iloc[0]
    eff_dir = (sub_vio[1]['EffectiveDirective']).iloc[0]
    docuri_1 = docuri.split("/")[0]
    docuri_2 = docuri.split("/")[2]
    same_or=docuri_1+"//"+docuri_2
    url_collect=[]
    for blocku in sub_vio[1]['BlockedUri']:
        blocku_or_1 = blocku.split("/")[0]
        if (blocku_or_1 == "https:" or blocku_or_1 == "http:"):
            blocku_or_2 = blocku.split("/")[2]
            same_or2=blocku_or_1+"//"+blocku_or_2
            url_collect.append(same_or2)
    #Deduplication
    url_collect=list(set(url_collect))
    if same_or in url_collect:
        url_collect.remove(same_or)  #do not consider self url
       # print("ohoh! self")
#{'child-src':0,'img-src': 0, 'font-src': 0, 'style-src-elem': 0, 'script-src-elem': 0, 'script-src': 0, 'script-src-attr': 0, 'style-src-attr': 0, 'frame-src': 0, 'connect-src': 0, 'media-src': 0, 'worker-src': 0, 'object-src': 0, 'prefetch-src': 0, 'manifest-src': 0}
    if eff_dir=='img-src':
        img_url.append(len(url_collect))
    if eff_dir=='font-src':
        font_url.append(len(url_collect))
    if eff_dir=='style-src-elem':
        style_elem_url.append(len(url_collect))
    if eff_dir=='script-src-elem':
        script_elem_url.append(len(url_collect))
    if eff_dir=='connect-src':
        connect_url.append(len(url_collect))
    if eff_dir=='frame-src':
       frame_url.append(len(url_collect))



print(np.sum(style_elem_url))
print(np.sum(script_elem_url))
print(np.sum(img_url))
print(np.sum(connect_url))
print(np.sum(font_url))
print(np.sum(frame_url))
############################################################
home_page_nr=df_ncsp_all[df_ncsp_all['type']=="domain"]
himg_url=[]
hchild_url=[]
hfont_url=[]
hframe_url=[]
hstyle_elem_url=[]
hscript_elem_url=[]
hstyle_attr_url=[]
hscript_attr_url=[]
hconnect_url=[]
hwork_url=[]
hmedia_url=[]
hobject_url=[]
hprefetch_url=[]
hmanifest_url=[]
hscript_src_url=[]
print("violations of none_csp website home_pages", len(home_page_nr))
nopage=home_page_nr.groupby(['DocumentUri','EffectiveDirective'],as_index=False)
for sub_vio in nopage:
    docuri = (sub_vio[1]['DocumentUri']).iloc[0]
    eff_dir = (sub_vio[1]['EffectiveDirective']).iloc[0]
    docuri_1 = docuri.split("/")[0]
    docuri_2 = docuri.split("/")[2]
    same_or=docuri_1+"//"+docuri_2
    url_collect=[]
    for blocku in sub_vio[1]['BlockedUri']:
        blocku_or_1 = blocku.split("/")[0]
        if (blocku_or_1 == "https:" or blocku_or_1 == "http:"):
            blocku_or_2 = blocku.split("/")[2]
            same_or2=blocku_or_1+"//"+blocku_or_2
            url_collect.append(same_or2)
    #Deduplication
    url_collect=list(set(url_collect))
    #if same_or in url_collect:
    #   url_collect.remove(same_or)  #do not consider self url
       # print("ohoh! self")
#{'child-src':0,'img-src': 0, 'font-src': 0, 'style-src-elem': 0, 'script-src-elem': 0, 'script-src': 0, 'script-src-attr': 0, 'style-src-attr': 0, 'frame-src': 0, 'connect-src': 0, 'media-src': 0, 'worker-src': 0, 'object-src': 0, 'prefetch-src': 0, 'manifest-src': 0}
    if eff_dir=='img-src':
        himg_url.append(len(url_collect))
    if eff_dir=='font-src':
        hfont_url.append(len(url_collect))
    if eff_dir=='style-src-elem':
        hstyle_elem_url.append(len(url_collect))
    if eff_dir=='script-src-elem':
        hscript_elem_url.append(len(url_collect))
    if eff_dir=='style-src-attr':
        hstyle_attr_url.append(len(url_collect))
    if eff_dir=='script-src-attr':
        hscript_attr_url.append(len(url_collect))
    if eff_dir=='connect-src':
        hconnect_url.append(len(url_collect))
    if eff_dir=='frame-src':
        hframe_url.append(len(url_collect))
    if eff_dir == 'child-src':
        hchild_url.append(len(url_collect))
    if eff_dir=='script-src':
        hscript_src_url.append(len(url_collect))
    if eff_dir=='media-src':
        hmedia_url.append(len(url_collect))
    if eff_dir=='worker-src':
        hwork_url.append(len(url_collect))
    if eff_dir=='object-src':
        hobject_url.append(len(url_collect))
    if eff_dir=='prefetch-src':
        hprefetch_url.append(len(url_collect))
    if eff_dir=='manifest-src':
        hmanifest_url.append(len(url_collect))

print(np.sum(hstyle_elem_url))
print(np.sum(hscript_elem_url))
print(np.sum(himg_url))
print(np.sum(hconnect_url))
print(np.sum(hfont_url))
print(np.sum(hframe_url))
#################################website