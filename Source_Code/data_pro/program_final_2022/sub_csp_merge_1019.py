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
file_path="/Volumes/Elements/CSPNEW_DATA/"
#Datafolder=["CSPNEW02"]
Datafolder=["CSPNEW01","CSPNEW02","CSPNEW03","CSPNEW04","CSPNEW05-1","CSPNEW05-2","CSPNEW06","CSPNEW07-1","CSPNEW07-2","CSPNEW08-1","CSPNEW08-2","CSPNEW09-1","CSPNEW09-2","CSPNEW10-1","CSPNEW10-2"]
csp_dset=[]
enforcement_homepage_total=[]
reportonly_homepage_total=[]
data_set_len=0
whole_csp_set=[]
fo_num=0
for fo_num in range(1,16):
    file_name_con = "/Volumes/Elements/CSP_HOMEPAGE/df_concat_encsp"+str(fo_num)+".csv"
    csv_data = pd.read_csv(file_name_con, low_memory=False)
    df_con = pd.DataFrame(csv_data)
    df_con.drop_duplicates(["site_url_y","domain","cspmetaheader","cspheader","cspmetacon","cspcontent"], 'last', inplace=True)
    df_con["domain"]=df_con["domain"]+fo_num*10000
    whole_csp_set.append(df_con)
#select milticsp in a webpage:
df_whole_csp_set=pd.concat(whole_csp_set,ignore_index=True,axis=0)
df_whole_csp_set.to_csv(file_path+"/df_concat_encsp_all.csv")
print(len(df_whole_csp_set))


##################extract csp features
csp_dset=[]
all_directives = ["child-src", "connect-src", "default-src", "font-src", "frame-src", "img-src", "manifest-src",
                    "media-src", "object-src", "prefetch-src", "script-src", "style-src", "worker-src","style-src-elem",
                    "script-src-elem", "style-src-attr", "script-src-attr",
                    "base-uri","sandbox",
                    "form-action","frame-ancestors","navigate-to",
                    "block-all-mixed-content","upgrade-insecure-requests",
                    "trusted-types","require-sri-for", "require-trusted-types-for","plugin-types"#,"policy-definition" #additional directives
                  ]
print("all_directives:",len(all_directives))
special_process_directives=["block-all-mixed-content","upgrade-insecure-requests","plugin-types"]
sandbox=["sandbox"]
other_directives=["trusted-types","require-sri-for", "require-trusted-types-for"]

#dropped_csp=[]
i=0
n=0
df_csp_info = df_whole_csp_set
for ind, csp_rec in df_csp_info.iterrows():
    if csp_rec["cspheader"] == "designed_CSP_only":
        df_csp_info.at[ind, "cspcontent"] = "nxc"
    if csp_rec["cspmetaheader"] == "none_csp":
        df_csp_info.at[ind, "cspmetacon"] = "nxc"

#     #####csp deployed on enforced mode:
# for ind, csp_rec in df_csp_info.iterrows():
#         if csp_rec["cspheader"].lower() != "content-security-policy":
#             df_csp_info.at[ind, "cspcontent"] = "nxc"
#         if csp_rec["cspmetaheader"].lower() != "content-security-policy":
#             df_csp_info.at[ind, "cspmetacon"] = "nxc"

# df_csp_info=df_csp_info[(df_csp_info["cspcontent"]!="nxc") | (df_csp_info["cspmetacon"]!="nxc")]
# df_csp_info.to_csv(file_path + "/df_csp_cspmode_set.csv")
# print("total number of homepages deployed a CSP on enforced mode:", len(df_csp_info))

    ###extact CSP:
df_csp_info_mh=pd.DataFrame(columns=["Site","csp_con","domain","csp_mode","ctype","type"])
for index, row in df_csp_info.iterrows():
        if (row["ctype"] == "NCSP"):
            df_csp_info_mh = df_csp_info_mh.append({"Site": row["site_url_y"], "csp_con": "nxc", "domain": row["domain"],"csp_mode": "ncsp","type": row["type"],"ctype":row["ctype"]}, ignore_index=True)
        elif(row["ctype"] == "CSP"):
           if(row["cspmetacon"] != "nxc"):
            df_csp_info_mh = df_csp_info_mh.append({"Site": row["site_url_y"], "csp_con": row["cspmetacon"], "domain": row["domain"], "csp_mode": row["cspmetaheader"].lower(),"type": row["type"],"ctype":row["ctype"]}, ignore_index=True)
           if (row["cspcontent"] != "nxc"):
            df_csp_info_mh  = df_csp_info_mh .append({"Site": row["site_url_y"], "csp_con": row["cspcontent"],"domain": row["domain"],"csp_mode": row["cspheader"].lower(),"type": row["type"],"ctype":row["ctype"]}, ignore_index=True)
df_csp_info_mh.drop_duplicates(["Site","csp_con","domain","csp_mode"], 'last', inplace=True)
df_csp_info_mh.to_csv((file_path+"/df_csp_info_mh.csv"))
csp_dset.append(df_csp_info_mh)
print(file_path,"total number of extracted CSP :", len(df_csp_info_mh))

special_process_schemes=["https","http","data","blob","ws","wss"]
customized_scheme="customized_scheme"
#len(all_values):6+8+4+2(*./*)=20,all values generated features are 6+8+4+5=23
all_values = ["unsafe-inline", "unsafe-eval", "unsafe-hashes", "strict-dynamic", "self", "none","unsafe-allow-redirects","report-sample",
             "*","*.",
             "nonce-", "sha256-", "sha384-", "sha512-",
             "https","http","data","blob","ws","wss"
              ]
print("all values:",len(all_values))
special_process_values=["nonce-", "sha256-", "sha384-", "sha512-"]
sand_box_values=["allow-downloads","allow-downloads-without-user-activation","allow-forms","allow-modals",
                 "allow-orientation-lock","allow-same-origin","allow-scripts","allow-storage-access-by-user-activation",
                 "allow-top-navigation","allow-top-navigation-by-user-activation",
                 "allow-pointer-lock","allow-popups","allow-popups-to-escape-sandbox","allow-presentation"]
other_value=["script","style","allow-duplicates"]
################################################################################process CSP: no pre-statistic
#########generate_features:  (23+1)*21+3+15+8=530
for csp_subset in csp_dset:
    for dire in all_directives:
        if ((dire not in special_process_directives) and (dire not in other_directives) and (dire not in sandbox)):
          csp_subset[dire + "_customized_scheme"]=0 #add customized_scheme feature
          for val in all_values:
              if(val!="*."):
                csp_subset[dire + "_"+ val] = 0
              else:
                csp_subset[dire + "_" + val+"_sado"] = 0
                csp_subset[dire + "_" + val + "_exdo"] = 0
                csp_subset[dire + "_n_" + val + "_sado"] = 0
                csp_subset[dire + "_n_" + val + "_exdo"] = 0
        else:
            if(dire not in other_directives and (dire not in sandbox)):
               csp_subset[dire] = 0
    # require-sri-for script;
    # require-sri-for style;
    # require-sri-for script style;

    # require-trusted-types-for 'script'


    # Content-Security-Policy: trusted-types;
    # Content-Security-Policy: trusted-types 'none';
    # Content-Security-Policy: trusted-types <policyName>;
    # Content-Security-Policy: trusted-types <policyName> <policyName> 'allow-duplicates';

    # Content-Security-Policy: plugin-types <type>/<subtype> <type>/<subtype>;
    csp_subset["require-sri-for_script"]=0
    csp_subset["require-sri-for_style"] = 0
    csp_subset["require-sri-for_script_style"] = 0

    csp_subset["require-trusted-types-for_script"]=0

    csp_subset["trusted-types"] = 0
    csp_subset["trusted-types_none"] = 0
    csp_subset["trusted-types_policyname"] = 0
    csp_subset["trusted-types_allow-duplicates"]=0

    csp_subset["sandbox"] = 0
    for sval in sand_box_values:
        csp_subset["sandbox_" + sval] = 0
    print("feature_len:",len(csp_subset.columns))

    for index, row in csp_subset.iterrows():
        csp_rec=row["csp_con"].split(";")
        do_site = tldextract.extract(row['Site']).domain + "." + tldextract.extract(row['Site']).suffix
        # if (do_site == "robinhood.com"):
        #  print("dosite:",do_site)
        for cc in csp_rec:
            cc = cc + ";"
            cc_low=cc.lower()
            for dire in all_directives:
              if ( cc_low.find(dire) > -1 and (dire in special_process_directives)):
                  if (csp_subset.at[index, dire] == 0):
                        csp_subset.at[index, dire] = 1
              elif((cc_low.find(dire+" ") > -1 or cc_low.find(dire+";") > -1 ) and (dire in other_directives)):
                  if(dire=="require-sri-for"):
                      script_value=0
                      style_value=0
                      ccs = cc.split(" ")
                      for ccss in ccs:
                          if(ccss.find("script")>-1):
                              script_value=1
                          if (ccss.find("style") > -1):
                              style_value = 1
                      if(style_value==1 and script_value==1):
                          if (csp_subset.at[index, dire + "_script_style"] == 0):
                              csp_subset.at[index, dire + "_script_style"] = 1
                      elif(style_value==1 and script_value==0):
                          if (csp_subset.at[index, dire + "_style"] == 0):
                              csp_subset.at[index, dire + "_style"] = 1
                      elif(style_value==0 and script_value==1):
                          if (csp_subset.at[index, dire + "_script"] == 0):
                              csp_subset.at[index, dire + "_script"] = 1
                  elif (dire == "require-trusted-types-for"):
                      ccs = cc.split(" ")
                      for ccss in ccs:
                          if(ccss.find("script")>-1):
                              if (csp_subset.at[index, dire + "_script"] == 0):
                                  csp_subset.at[index, dire + "_script"] = 1
                  elif (dire == "trusted-types" and cc_low.find("require-trusted-types-for") == -1):
                      policyname = 0
                      ccs = cc.split(" ")
                      for ccss in ccs:
                          if (ccss.find("none") > -1):
                              if (csp_subset.at[index, dire + "_none"] == 0):
                                  csp_subset.at[index, dire + "_none"] = 1
                          elif(ccss.find("allow-duplicates") > -1):
                              if (csp_subset.at[index, dire + "_allow-duplicates"] == 0):
                                  csp_subset.at[index, dire + "_allow-duplicates"] = 1
                          else:
                              if(len(ccss)>0 and ccss.find("none")==-1 and ccss.find("allow-duplicates")==-1):
                                  policyname=1
                      if(policyname==0):
                          if (csp_subset.at[index, dire] == 0):
                           csp_subset.at[index, dire] = 1
                      else:
                          if (csp_subset.at[index, dire + "_policyname"] == 0):
                           csp_subset.at[index, dire + "_policyname"] = 1
              elif(cc_low.find(dire+" ") > -1 and (dire in sandbox)):
                  ccs = cc.split(" ")
                  sval = 0
                  for ccss in ccs:
                      for va in sand_box_values:
                          if (ccss.find(va) > -1 and va!="allow-top-navigation" and va!="allow-popups" and va!="allow-downloads"):
                              sval=1
                              if (csp_subset.at[index, dire + "_" + va] == 0):
                                  csp_subset.at[index, dire + "_" + va] = 1
                          elif(ccss.find(va) > -1 and va=="allow-top-navigation" and ccss.find("allow-top-navigation-by-user-activation")== -1):
                              sval = 1
                              if (csp_subset.at[index, dire + "_" + va] == 0):
                                  csp_subset.at[index, dire + "_" + va] = 1
                          elif (ccss.find(va) > -1 and va == "allow-popups" and ccss.find("allow-popups-to-escape-sandbox") == -1):
                              sval = 1
                              if (csp_subset.at[index, dire + "_" + va] == 0):
                                  csp_subset.at[index, dire + "_" + va] = 1
                          elif (ccss.find(va) > -1 and va == "allow-downloads" and ccss.find("allow-downloads-without-user-activation") == -1):
                              sval = 1
                              if (csp_subset.at[index, dire + "_" + va] == 0):
                                  csp_subset.at[index, dire + "_" + va] = 1

                  if(sval==0):
                          if (csp_subset.at[index, dire] == 0):
                              csp_subset.at[index, dire] = 1
              elif(cc_low.find(dire+" ") > -1):
                        #print("dire:",dire)
                        ccs = cc.split(" ")
                        #print("ccs:",ccs)
                        #if ((ccs not in special_process_values) and (ccs not in special_process_schemes)):
                        for ccss in ccs:
                            ccss2=ccss+" "
                            if ((ccss2.find(": ")>-1 or ccss2.find(":;")>-1
                                or ccss2.find(":* ")>-1 or ccss2.find( ":*;")>-1
                                or ccss2.find(":// ")>-1 or ccss2.find( "://;")>-1
                                or ccss2.find( "://* ")>-1 or ccss2.find("://*;")>-1)
                            and
                            (tldextract.extract(ccss.replace(";","")).suffix=="")
                            ):
                                ccss2=ccss2.replace(" ","")
                                ccss2=ccss2.replace(";","")
                                ccss2=ccss2.replace("*","")
                                ccss2=ccss2.replace("//","")
                                ccss2 = ccss2.replace(":", "")
                                if(ccss2 not in special_process_schemes):
                                    if (csp_subset.at[index, dire + "_" + customized_scheme] == 0):
                                        csp_subset.at[index, dire + "_" + customized_scheme] = 1
                                        # print("ccss2:",ccss2)

                            for d_va in all_values:
                                if((d_va in special_process_schemes)):
                                    if (ccss==d_va or ccss==(d_va + ";")
                                        or ccss==(d_va + ":") or ccss==(d_va + ":;")
                                        or ccss==(d_va + ":*") or ccss==(d_va + ":*;")
                                        or ccss==(d_va + "://") or ccss==(d_va + "://;")
                                        or ccss==(d_va + "://*") or ccss==(d_va + "://*;")
                                    ):
                                        if(csp_subset.at[index, dire + "_" + d_va] ==0):
                                           csp_subset.at[index, dire + "_" + d_va] = 1
                                elif(d_va in special_process_values):
                                      if(ccss.find(d_va) > -1 and (tldextract.extract(ccss.replace(";","")).suffix=="")):
                                          if (csp_subset.at[index, dire + "_" + d_va] == 0):
                                               csp_subset.at[index, dire + "_" + d_va] = 1
                                elif(d_va=="*."):
                                        ccss_0=ccss.replace(";","")
                                        ccss_0=ccss_0.replace("*","")
                                        ccss_0=ccss_0.replace("//","")
                                        ccss_0 = ccss_0.replace(":", "")

                                        if(tldextract.extract(ccss.replace(";","")).suffix!="" and (ccss_0 not in special_process_schemes)):
                                        #if(tldextract.extract(ccss.replace(";","")).suffix!="" and ((tldextract.extract(ccss.replace(";","")).suffix not in special_process_schemes) or
                                         #(ccss.replace(":", "").replace(";", "") not in special_process_schemes))
                                        #):
                                          # print("ccss:", ccss)
                                          #if(tldextract.extract(ccss.replace(";","")).suffix=="data"):
                                          #print("ccss0:", ccss_0)
                                          if(ccss.find("*.")>-1 or ccss.find(".*")>-1 or ccss.find("/*")>-1 or ccss.find(":*")>-1):


                                            ccss_new = ccss.replace(";", "")
                                            csite = tldextract.extract(ccss_new).domain + "." + tldextract.extract(ccss_new).suffix
                                        #remove localhost and 127.0.0.1
                                            if (csite == do_site):
                                                if (csp_subset.at[index, dire + "_" + d_va + "_sado"] == 0):
                                                       csp_subset.at[index,dire + "_" + d_va + "_sado"] = 1
                                            else:
                                                if (csp_subset.at[index, dire + "_" + d_va + "_exdo"] == 0):
                                                    csp_subset.at[index,dire + "_" + d_va + "_exdo"] = 1
                                          else:
                                              ccss_new = ccss.replace(";", "")
                                              csite = tldextract.extract(ccss_new).domain + "." + tldextract.extract(
                                                  ccss_new).suffix
                                              if (csite == do_site):
                                                  if (csp_subset.at[index, dire + "_n_" + d_va + "_sado"] == 0):
                                                      csp_subset.at[index, dire + "_n_" + d_va + "_sado"] = 1

                                              else:
                                                  if (csp_subset.at[index, dire + "_n_" + d_va + "_exdo"] == 0):
                                                      csp_subset.at[index, dire + "_n_" + d_va + "_exdo"] = 1
                                else:
                                    if(((" "+ccss+" ").find(" "+d_va+" ")>-1 or (" "+ccss+" ").find(" "+d_va+";")>-1) and tldextract.extract(
                                                  ccss.replace(";","")).suffix=="" ):
                                        if(csp_subset.at[index, dire + "_" + d_va] ==0):
                                           csp_subset.at[index, dire + "_" + d_va] = 1
    csp_subset.to_csv((file_path+"df_csp_normfeature"+str(n)+".csv"))
    n=n+1

    # for index, row in csp_subset.iterrows():
    #     for dire in all_directives:
    #         if(row[dire + "_ex_do"]>0):
    #             csp_subset.at[index, dire + "_ex_do"] =1
    # # df_csp_info.to_csv((file_name+"/df_csp_bin_feature.csv"))
    i=i+1

# print("dropped_csp:")
# print(dropped_csp)
#     for col in csp_subset.columns:
#         print(col)
#     csp_group=csp_subset.groupby(["domain","Site","csp_mode"])
#     for gn,g in csp_group:
#           if(len(g)>1):
#               d = g.iloc[0]
#               inde=g.index[0]
#               print("inde",inde)
#               for col in csp_subset.columns:
#                 if (col != "Site" and col != "csp_con" and col != "domain" and col != "csp_mode"):
#                   ini_val=d[col]
#                   for index, de in g.iterrows():
#                       ini_val=ini_val+de[col]
#                   if(ini_val>1):
#                      ini_val=1
#                   csp_subset.at[inde,col]=ini_val      #print(d[col])
#               for index, de in g.iterrows():
#                   if(index!=inde):
#                     csp_subset.drop(index=index,inplace=True)
#     print(len(list(set(csp_subset["domain"]))))
#     print(len(csp_subset))
#     csp_subset.to_csv(file_path+"/df_merge_csp_middle"+str(i)+".csv")
#
#     cspgroup = csp_subset.groupby(["domain", "Site"])
#     for gn, g in csp_group:
#         if(len(g)>1 and ("content-security-policy" in g["csp_mode"]) and ("content-security-policy-report-only" in g["csp_mode"])):
#              for index, dele in g.iterrows():
#                  if(dele["csp_mode"]=="content-security-policy-report-only"):
#                      csp_subset.drop(index=index, inplace=True)
#     #######select CSP
#     print(len(csp_subset))
#     print("total_page:",len(list(set(csp_subset["Site"]))))
#     c_group=csp_subset.groupby("domain")
#     no_sub=[]#no subpages, only home page
#     all_sub_csp=[]#all subpages deployed a CSP
#     not_all_enfor=[]
#     all_sub_en_mode=[]#all subpage deploy an enforcement mode csp.
#     for g_name, g in c_group:
#         domain=g.iloc[0]["domain"]
#         if(len(g)==1 and ("domain" in g["type"])):
#             no_sub.append(domain)
#         else:
#             #subpage's number
#             sub_num=len(g)
#             not_enfor=0
#             for ind, sub_page in g.iterrows():
#                 if(sub_page["csp_mode"]!="content-security-policy"):
#                     not_enfor=1
#                     csp_subset.drop(index=ind, inplace=True)
#             if(not_enfor==1):
#                 not_all_enfor.append(domain)
#             if(not_enfor==0):
#                 all_sub_en_mode.append(domain)
#     print("all_sub_en_mode:",len(all_sub_en_mode))
#     print("not_all_enfor:", len(not_all_enfor))
#     csp_enforce=csp_subset[csp_subset["csp_mode"]=="content-security-policy"]
#     scsp_all =csp_enforce[csp_enforce["domain"].isin(all_sub_en_mode)]
#     scsp_part=csp_enforce[csp_enforce["domain"].isin(not_all_enfor)]
#
#     not_same=[]
#     same=[]
#     con_same=[]
#     c_group=scsp_part.groupby("domain")
#     for g_name, g in c_group:
#         domain=g.iloc[0]["domain"]
#         dom=g.iloc[0]
#         nsame=0
#         csame=0
#         for ind, sub_page in g.iterrows():
#             if(sub_page["csp_con"]!=dom["csp_con"]):
#                 csame=1
#             for col in scsp_part.columns:
#                 if(col!="Site" and col!=""):
#                     if(sub_page[col]!=dom[col]):
#                         nsame=1
#         if(nsame==1):
#             not_same.append(domain)
#         if(nsame==0):
#             same.append(domain)
#         if(csame==0):
#             con_same.append(domain)
#     print("not_same len:",len(not_same))
#     print("same len:", len(same))
#     print("con_same len:", len(con_same))
#
#
#     not_same=[]
#     same=[]
#     con_same=[]
#     c_group=scsp_all.groupby("domain")
#     for g_name, g in c_group:
#         domain=g.iloc[0]["domain"]
#         dom=g.iloc[0]
#         nsame=0
#         csame=0
#         for ind, sub_page in g.iterrows():
#             if(sub_page["csp_con"]!=dom["csp_con"]):
#                 csame=1
#             for col in scsp_part.columns:
#                 if(col!="Site" and col!=""):
#                     if(sub_page[col]!=dom[col]):
#                         nsame=1
#         if(nsame==1):
#             not_same.append(domain)
#         if(nsame==0):
#             same.append(domain)
#         if(csame==0):
#             con_same.append(domain)
#     print("not_same len:",len(not_same))
#     print("same len:", len(same))
#     print("con_same len:", len(con_same))

