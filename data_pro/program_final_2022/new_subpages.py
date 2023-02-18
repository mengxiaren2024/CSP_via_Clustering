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
site_has_sub=[]
site_csp_list=[]
site_suball_CSP_en_list=[]
site_subpart_CSP_list=[]
site_only_homepage_list=[]
site_no_sub_CSP_list=[]
j=0
all_len=0
all_len3=0
fold_num=0
for dafolder in Datafolder:
    fold_num=fold_num+1
    file_name=file_path+dafolder
    file_name_con = file_name+"/df_concat_sub_3.csv"
    csv_data = pd.read_csv(file_name_con, low_memory=False)
    df_con = pd.DataFrame(csv_data)

    file_name_con2 = "/Volumes/Elements/CSP_HOMEPAGE/df_merge_csp.csv"
    csv_data2 = pd.read_csv(file_name_con2, low_memory=False)
    df_con2 = pd.DataFrame(csv_data2)
    domains=df_con2[(df_con2["domain"]-10000*fold_num<10000) & (df_con2["domain"]-10000*fold_num>0)]["domain"]-10000*fold_num
    all_len3=all_len3+len(list(set(domains)))
    df_con=df_con[df_con["domain"].isin(domains)]
    df_con.to_csv("/Volumes/Elements/CSP_HOMEPAGE/df_concat_encsp"+str(fold_num)+".csv")
    all_len=all_len+len(set(df_con["domain"]))

# print("len:",all_len)
# print("len:",all_len3)



    # all_group = df_con.groupby("domain")
    # print("groyp size:",all_group.size)
    # print("webpages with a CSPs:", len(df_con))
    # for gname, allg in all_group:
    #     #print(allg["ctype"])
    #     dmain=""
    #     doma=-1
    #     n_ncsp=0
    #     n_csp=0
    #     do_csp=0
    #     enforce=0
    #     for index, al in allg.iterrows():
    #         if (al["type"] == "domain"):
    #            dmain=al["site_url_y"]
    #            doma=al["domain"]+j*10000
    #            if(al["ctype"]=="CSP"):
    #                do_csp=1
    #         if(al["ctype"]=="NCSP"):
    #             n_ncsp=n_ncsp+1
    #         if(al["ctype"]=="CSP"):
    #             n_csp=n_csp+1
    #         if(al["cspmetaheader"].lower()=="CSP"):
    #             enforce=enforce+1
    #     if(do_csp==1):
    #         site_csp_list.append(doma)
    #     if (("sub_domain" not in set(allg["type"])) and do_csp==1):
    #         site_no_sub_CSP_list.append(doma)
    #     if (("sub_domain" in set(allg["type"])) and ("domain" in set(allg["type"]))):
    #        site_has_sub.append(dmain)
    #        if("NCSP" not in list(set(allg["ctype"])) and enforce==len(allg)):
    #          # sub_all_csp.append(dmain)
    #          site_suball_CSP_en_list.append(doma)
    #        elif((do_csp==1) and (n_csp>1) and (n_ncsp>0)):
    #          site_subpart_CSP_list.append(doma)
    #        elif((do_csp==1) and (n_csp==1)):
    #          site_only_homepage_list.append(doma)
    #        elif(do_csp==1):
    #            print("domain:",doma)
    # print(dafolder + " has subpages:", len(list(set(site_has_sub))))
    # print(dafolder+" all suball CSP:",len(sub_all_csp))
    #print(dafolder + " homepage without subpages:", len(site_no_sub_home_CSP))
    ################################################################filter website that do not has a subpage
    # domain_set=set(df_csp_info["domain"])
    # df_new=df_csp_info[df_csp_info["type"] == "domain"]
    # homepage_set=set(df_new["domain"])
    # print("homepages:",len(homepage_set))
    # print("number of CSP homepages:",len(list(set(homepage_set))))
    # # print("wrong:",list(set(df_csp_info[df_csp_info["domain"].isin(homepage_set)])))
    # print("number of webpages deployed a CSPs:",len(df_csp_info[df_csp_info["domain"].isin(homepage_set)]))
    # # print("len before:",len(df_csp_info))
    # for ind, row in df_csp_info.iterrows():
    #     if (row["domain"] not in homepage_set):
    #         df_csp_info.drop(index=ind, inplace=True)
    # print("len after home delete:",len( df_csp_info))
    # # print("len after home delete set:", len(list(set(df_csp_info["domain"]))))
    # delete_domain=[]
    # all_delete=site_only_homepage_list+site_no_sub_CSP_list
    # for el in all_delete:
    #     if( (el-j*10000)>0 and (el-j*10000)<10000):
    #        delete_domain.append(el-j*10000)
    # print("delete domain:", len(delete_domain))
    #
    # j=j+1
    #
    # for ind, row in df_csp_info.iterrows():
    #   if (row["domain"] in delete_domain):
    #     df_csp_info.drop(index=ind, inplace=True)
    # print("len after deltet:", len(df_csp_info))
    # df_csp_info.to_csv(file_name + "/df_concat_sub_4.csv")

#
#
#
#
#
#     ###############################################################
#
#     for ind, csp_rec in df_csp_info.iterrows():
#         if csp_rec["cspheader"] == "designed_CSP_only":
#             df_csp_info.at[ind, "cspcontent"] = "nxc"
#         if csp_rec["cspmetaheader"] == "none_csp":
#             df_csp_info.at[ind, "cspmetacon"] = "nxc"
#
#     #####csp deployed on enforced mode:
#     # for ind, csp_rec in df_csp_info.iterrows():
#     #     if csp_rec["cspheader"].lower() != "content-security-policy":
#     #         df_csp_info.at[ind, "cspcontent"] = "nxc"
#     #     if csp_rec["cspmetaheader"].lower() != "content-security-policy":
#     #         df_csp_info.at[ind, "cspmetacon"] = "nxc"
#
#     df_csp_info=df_csp_info[(df_csp_info["cspcontent"]!="nxc") | (df_csp_info["cspmetacon"]!="nxc")]
#     df_csp_info.to_csv(file_name + "/df_csp_cspmode_set_sub.csv")
#     print("total number of webpages deployed a CSP:", len(df_csp_info))
#
#     ###extact CSP:
#     df_csp_info_mh=pd.DataFrame(columns=["Site","csp_con","type","domain","cspheader","cspmetaheader"])
#     for index, row in df_csp_info.iterrows():
#
#         if (row["cspmetacon"] != "nxc"):
#             df_csp_info_mh = df_csp_info_mh.append({"Site": row["site_url_y"], "csp_con": row["cspmetacon"],"type":row["type"],"domain":row["domain"],"cspheader":"N","cspmetaheader":row["cspmetaheader"]}, ignore_index=True)
#         if (row["cspcontent"] != "nxc"):
#             df_csp_info_mh  = df_csp_info_mh .append({"Site": row["site_url_y"], "csp_con": row["cspcontent"],"type":row["type"],"domain":row["domain"],"cspheader":row["cspheader"],"cspmetaheader":'N'}, ignore_index=True)
#     df_csp_info_mh.drop_duplicates(["Site","csp_con"], 'last', inplace=True)
#     df_csp_info_mh.to_csv((file_name+"/df_csp_info_mh_sub.csv"))
#     csp_dset.append(df_csp_info_mh)
#     print(file_name,"total number of extracted CSP :", len(df_csp_info_mh))
#
# print(" all CSP sites:", len(site_csp_list))
# print(" all suball CSP:", len(site_suball_CSP_list))
# print(" all subpart CSP:",len(site_subpart_CSP_list))
# print(" all homepageonly CSP with sub:",len(site_only_homepage_list))
# print(" all only homepage CSP without sub:",len(site_no_sub_CSP_list))
# special_process_schemes=["https","http","data","blob","ws","wss"]
# customized_scheme="customized_scheme"
# #len(all_values):6+8+4+2(*./*)=20,all values generated features are 6+8+4+5=23
# all_values = ["unsafe-inline", "unsafe-eval", "unsafe-hashes", "strict-dynamic", "self", "none","unsafe-allow-redirects","report-sample",
#              "*","*.",
#              "nonce-", "sha256-", "sha384-", "sha512-",
#              "https","http","data","blob","ws","wss"
#               ]
# print("all values:",len(all_values))
# special_process_values=["nonce-", "sha256-", "sha384-", "sha512-"]
# sand_box_values=["allow-downloads","allow-downloads-without-user-activation","allow-forms","allow-modals",
#                  "allow-orientation-lock","allow-same-origin","allow-scripts","allow-storage-access-by-user-activation",
#                  "allow-top-navigation","allow-top-navigation-by-user-activation",
#                  "allow-pointer-lock","allow-popups","allow-popups-to-escape-sandbox","allow-presentation"]
# other_value=["script","style","allow-duplicates"]
# ################################################################################process CSP: no pre-statistic
# #########generate_features:  (23+1)*21+3+15+8=530
# for csp_subset in csp_dset:
#     for dire in all_directives:
#         if ((dire not in special_process_directives) and (dire not in other_directives) and (dire not in sandbox)):
#           csp_subset[dire + "_customized_scheme"]=0 #add customized_scheme feature
#           for val in all_values:
#               if(val!="*."):
#                 csp_subset[dire + "_"+ val] = 0
#               else:
#                 csp_subset[dire + "_" + val+"_sado"] = 0
#                 csp_subset[dire + "_" + val + "_exdo"] = 0
#                 csp_subset[dire + "_n_" + val + "_sado"] = 0
#                 csp_subset[dire + "_n_" + val + "_exdo"] = 0
#         else:
#             if(dire not in other_directives and (dire not in sandbox)):
#                csp_subset[dire] = 0
#     # require-sri-for script;
#     # require-sri-for style;
#     # require-sri-for script style;
#
#     # require-trusted-types-for 'script'
#
#
#     # Content-Security-Policy: trusted-types;
#     # Content-Security-Policy: trusted-types 'none';
#     # Content-Security-Policy: trusted-types <policyName>;
#     # Content-Security-Policy: trusted-types <policyName> <policyName> 'allow-duplicates';
#
#     # Content-Security-Policy: plugin-types <type>/<subtype> <type>/<subtype>;
#     csp_subset["require-sri-for_script"]=0
#     csp_subset["require-sri-for_style"] = 0
#     csp_subset["require-sri-for_script_style"] = 0
#
#     csp_subset["require-trusted-types-for_script"]=0
#
#     csp_subset["trusted-types"] = 0
#     csp_subset["trusted-types_none"] = 0
#     csp_subset["trusted-types_policyname"] = 0
#     csp_subset["trusted-types_allow-duplicates"]=0
#
#     csp_subset["sandbox"] = 0
#     for sval in sand_box_values:
#         csp_subset["sandbox_" + sval] = 0
#     print("feature_len:",len(csp_subset.columns))
#
#     for index, row in csp_subset.iterrows():
#         csp_rec=row["csp_con"].split(";")
#         do_site = tldextract.extract(row['Site']).domain + "." + tldextract.extract(row['Site']).suffix
#         # if (do_site == "robinhood.com"):
#         #  print("dosite:",do_site)
#         for cc in csp_rec:
#             cc = cc + ";"
#             cc_low=cc.lower()
#             for dire in all_directives:
#               if ( cc_low.find(dire) > -1 and (dire in special_process_directives)):
#                   if (csp_subset.at[index, dire] == 0):
#                         csp_subset.at[index, dire] = 1
#               elif((cc_low.find(dire+" ") > -1 or cc_low.find(dire+";") > -1 ) and (dire in other_directives)):
#                   if(dire=="require-sri-for"):
#                       script_value=0
#                       style_value=0
#                       ccs = cc.split(" ")
#                       for ccss in ccs:
#                           if(ccss.find("script")>-1):
#                               script_value=1
#                           if (ccss.find("style") > -1):
#                               style_value = 1
#                       if(style_value==1 and script_value==1):
#                           if (csp_subset.at[index, dire + "_script_style"] == 0):
#                               csp_subset.at[index, dire + "_script_style"] = 1
#                       elif(style_value==1 and script_value==0):
#                           if (csp_subset.at[index, dire + "_style"] == 0):
#                               csp_subset.at[index, dire + "_style"] = 1
#                       elif(style_value==0 and script_value==1):
#                           if (csp_subset.at[index, dire + "_script"] == 0):
#                               csp_subset.at[index, dire + "_script"] = 1
#                   elif (dire == "require-trusted-types-for"):
#                       ccs = cc.split(" ")
#                       for ccss in ccs:
#                           if(ccss.find("script")>-1):
#                               if (csp_subset.at[index, dire + "_script"] == 0):
#                                   csp_subset.at[index, dire + "_script"] = 1
#                   elif (dire == "trusted-types" and cc_low.find("require-trusted-types-for") == -1):
#                       policyname = 0
#                       ccs = cc.split(" ")
#                       for ccss in ccs:
#                           if (ccss.find("none") > -1):
#                               if (csp_subset.at[index, dire + "_none"] == 0):
#                                   csp_subset.at[index, dire + "_none"] = 1
#                           elif(ccss.find("allow-duplicates") > -1):
#                               if (csp_subset.at[index, dire + "_allow-duplicates"] == 0):
#                                   csp_subset.at[index, dire + "_allow-duplicates"] = 1
#                           else:
#                               if(len(ccss)>0 and ccss.find("none")==-1 and ccss.find("allow-duplicates")==-1):
#                                   policyname=1
#                       if(policyname==0):
#                           if (csp_subset.at[index, dire] == 0):
#                            csp_subset.at[index, dire] = 1
#                       else:
#                           if (csp_subset.at[index, dire + "_policyname"] == 0):
#                            csp_subset.at[index, dire + "_policyname"] = 1
#               elif(cc_low.find(dire+" ") > -1 and (dire in sandbox)):
#                   ccs = cc.split(" ")
#                   sval = 0
#                   for ccss in ccs:
#                       for va in sand_box_values:
#                           if (ccss.find(va) > -1 and va!="allow-top-navigation" and va!="allow-popups" and va!="allow-downloads"):
#                               sval=1
#                               if (csp_subset.at[index, dire + "_" + va] == 0):
#                                   csp_subset.at[index, dire + "_" + va] = 1
#                           elif(ccss.find(va) > -1 and va=="allow-top-navigation" and ccss.find("allow-top-navigation-by-user-activation")== -1):
#                               sval = 1
#                               if (csp_subset.at[index, dire + "_" + va] == 0):
#                                   csp_subset.at[index, dire + "_" + va] = 1
#                           elif (ccss.find(va) > -1 and va == "allow-popups" and ccss.find("allow-popups-to-escape-sandbox") == -1):
#                               sval = 1
#                               if (csp_subset.at[index, dire + "_" + va] == 0):
#                                   csp_subset.at[index, dire + "_" + va] = 1
#                           elif (ccss.find(va) > -1 and va == "allow-downloads" and ccss.find("allow-downloads-without-user-activation") == -1):
#                               sval = 1
#                               if (csp_subset.at[index, dire + "_" + va] == 0):
#                                   csp_subset.at[index, dire + "_" + va] = 1
#
#                   if(sval==0):
#                           if (csp_subset.at[index, dire] == 0):
#                               csp_subset.at[index, dire] = 1
#               elif(cc_low.find(dire+" ") > -1):
#                         #print("dire:",dire)
#                         ccs = cc.split(" ")
#                         #print("ccs:",ccs)
#                         #if ((ccs not in special_process_values) and (ccs not in special_process_schemes)):
#                         for ccss in ccs:
#                             ccss2=ccss+" "
#                             if ((ccss2.find(": ")>-1 or ccss2.find(":;")>-1
#                                 or ccss2.find(":* ")>-1 or ccss2.find( ":*;")>-1
#                                 or ccss2.find(":// ")>-1 or ccss2.find( "://;")>-1
#                                 or ccss2.find( "://* ")>-1 or ccss2.find("://*;")>-1)
#                             and
#                             (tldextract.extract(ccss.replace(";","")).suffix=="")
#                             ):
#                                 ccss2=ccss2.replace(" ","")
#                                 ccss2=ccss2.replace(";","")
#                                 ccss2=ccss2.replace("*","")
#                                 ccss2=ccss2.replace("//","")
#                                 ccss2 = ccss2.replace(":", "")
#                                 if(ccss2 not in special_process_schemes):
#                                     if (csp_subset.at[index, dire + "_" + customized_scheme] == 0):
#                                         csp_subset.at[index, dire + "_" + customized_scheme] = 1
#                                         # print("ccss2:",ccss2)
#
#                             for d_va in all_values:
#                                 if((d_va in special_process_schemes)):
#                                     if (ccss==d_va or ccss==(d_va + ";")
#                                         or ccss==(d_va + ":") or ccss==(d_va + ":;")
#                                         or ccss==(d_va + ":*") or ccss==(d_va + ":*;")
#                                         or ccss==(d_va + "://") or ccss==(d_va + "://;")
#                                         or ccss==(d_va + "://*") or ccss==(d_va + "://*;")
#                                     ):
#                                         if(csp_subset.at[index, dire + "_" + d_va] ==0):
#                                            csp_subset.at[index, dire + "_" + d_va] = 1
#                                 elif(d_va in special_process_values):
#                                       if(ccss.find(d_va) > -1 and (tldextract.extract(ccss.replace(";","")).suffix=="")):
#                                           if (csp_subset.at[index, dire + "_" + d_va] == 0):
#                                                csp_subset.at[index, dire + "_" + d_va] = 1
#                                 elif(d_va=="*."):
#                                         ccss_0=ccss.replace(";","")
#                                         ccss_0=ccss_0.replace("*","")
#                                         ccss_0=ccss_0.replace("//","")
#                                         ccss_0 = ccss_0.replace(":", "")
#
#                                         if(tldextract.extract(ccss.replace(";","")).suffix!="" and (ccss_0 not in special_process_schemes)):
#                                         #if(tldextract.extract(ccss.replace(";","")).suffix!="" and ((tldextract.extract(ccss.replace(";","")).suffix not in special_process_schemes) or
#                                          #(ccss.replace(":", "").replace(";", "") not in special_process_schemes))
#                                         #):
#                                           # print("ccss:", ccss)
#                                           #if(tldextract.extract(ccss.replace(";","")).suffix=="data"):
#                                           #print("ccss0:", ccss_0)
#                                           if(ccss.find("*.")>-1 or ccss.find(".*")>-1 or ccss.find("/*")>-1 or ccss.find(":*")>-1):
#
#
#                                             ccss_new = ccss.replace(";", "")
#                                             csite = tldextract.extract(ccss_new).domain + "." + tldextract.extract(ccss_new).suffix
#                                         #remove localhost and 127.0.0.1
#                                             if (csite == do_site):
#                                                 if (csp_subset.at[index, dire + "_" + d_va + "_sado"] == 0):
#                                                        csp_subset.at[index,dire + "_" + d_va + "_sado"] = 1
#                                             else:
#                                                 if (csp_subset.at[index, dire + "_" + d_va + "_exdo"] == 0):
#                                                     csp_subset.at[index,dire + "_" + d_va + "_exdo"] = 1
#                                           else:
#                                               ccss_new = ccss.replace(";", "")
#                                               csite = tldextract.extract(ccss_new).domain + "." + tldextract.extract(
#                                                   ccss_new).suffix
#                                               if (csite == do_site):
#                                                   if (csp_subset.at[index, dire + "_n_" + d_va + "_sado"] == 0):
#                                                       csp_subset.at[index, dire + "_n_" + d_va + "_sado"] = 1
#
#                                               else:
#                                                   if (csp_subset.at[index, dire + "_n_" + d_va + "_exdo"] == 0):
#                                                       csp_subset.at[index, dire + "_n_" + d_va + "_exdo"] = 1
#                                 else:
#                                     if(((" "+ccss+" ").find(" "+d_va+" ")>-1 or (" "+ccss+" ").find(" "+d_va+";")>-1) and tldextract.extract(
#                                                   ccss.replace(";","")).suffix=="" ):
#                                         if(csp_subset.at[index, dire + "_" + d_va] ==0):
#                                            csp_subset.at[index, dire + "_" + d_va] = 1
#     csp_subset.to_csv((file_path+"/CSP_CLUSTER/df_csp_normfeature_sub_"+str(n)+".csv"))
#     n=n+1

##############################################################################################
# file_path="/Volumes/Elements/CSPNEW_DATA/CSP_CLUSTER"
# csp_concat_vec_set=[]
# site_suball_CSP_list=[57, 112, 179, 320, 464, 498, 509, 520, 531, 564, 623, 645, 656, 856, 900, 930, 996, 1074, 1093, 1116, 1138, 1240, 1294, 1336, 1460, 1471, 1528, 1649, 1674, 1696, 1718, 1751, 1784, 1839, 1935, 2066, 2154, 2187, 2242, 2286, 2462, 2484, 2572, 2661, 2672, 2748, 2782, 2917, 2947, 2993, 3031, 3205, 3227, 3278, 3355, 3409, 3432, 3575, 3597, 3673, 3696, 3728, 3776, 3831, 3843, 3866, 3877, 3911, 3944, 4034, 4045, 4145, 4255, 4300, 4334, 4376, 4402, 4484, 4575, 4640, 4659, 4681, 4705, 4749, 4844, 4877, 5105, 5149, 5160, 5162, 5279, 5290, 5356, 5517, 5616, 5843, 5973, 6512, 6571, 6732, 6754, 6776, 6879, 6913, 6955, 7057, 7112, 7189, 7200, 7222, 7269, 7280, 7447, 7593, 7627, 7718, 7796, 7850, 7883, 8014, 8036, 8111, 8133, 8144, 8205, 8355, 8366, 8465, 8569, 10001, 10118, 10129, 10240, 10273, 10328, 10527, 10605, 10927, 11004, 11070, 11092, 11375, 11461, 11588, 11675, 12004, 12092, 12125, 12237, 12365, 12580, 12603, 12626, 12637, 12671, 12706, 12752, 12787, 12866, 12868, 12971, 13247, 13280, 13464, 13595, 13618, 13718, 13751, 13818, 13942, 14033, 14339, 14350, 14372, 14416, 14433, 14481, 14540, 14621, 14642, 14885, 14929, 14962, 14995, 15006, 15106, 15162, 15209, 15265, 15320, 15332, 15365, 15663, 15727, 15805, 15816, 15927, 16035, 16112, 16145, 16221, 16280, 16291, 16516, 16532, 16555, 16656, 16724, 16768, 16803, 17022, 17033, 17121, 17164, 17208, 17264, 17415, 17449, 17583, 17685, 17751, 17787, 17809, 17875, 17900, 18109, 18191, 18235, 18246, 18257, 18293, 18346, 18369, 18393, 18538, 18612, 20012, 20047, 20116, 20355, 20409, 20442, 20522, 20590, 20608, 20681, 20726, 20737, 20748, 20759, 20836, 20859, 20903, 21102, 21206, 21250, 21365, 21435, 21459, 21624, 21829, 21884, 21966, 22166, 22208, 22265, 22504, 22526, 22671, 22748, 22825, 22927, 22938, 22972, 22978, 22989, 23033, 23045, 23090, 23235, 23462, 23473, 23505, 23583, 23626, 23667, 23747, 23780, 23871, 23882, 23893, 23944, 24019, 24030, 24087, 24147, 24192, 24236, 24303, 24369, 24402, 24413, 24468, 24637, 24671, 25146, 25319, 25386, 25419, 25538, 25560, 25793, 25863, 25944, 25979, 26191, 26359, 26381, 26460, 26633, 26786, 26819, 27000, 27296, 27326, 27415, 27437, 27522, 27572, 27705, 28017, 28085, 28131, 28165, 28401, 28454, 28636, 28648, 28681, 28906, 30109, 30167, 30255, 30277, 30325, 30370, 30426, 30484, 30539, 30572, 30652, 31010, 31024, 31489, 31539, 31606, 31788, 31810, 31887, 31910, 31959, 31972, 32177, 32278, 32359, 32371, 32449, 32529, 32573, 32617, 32672, 32810, 32912, 33069, 33136, 33348, 33359, 33415, 33542, 33623, 33849, 33868, 34193, 34235, 34290, 34368, 34448, 34514, 34536, 34673, 34706, 34730, 34741, 35048, 35150, 35238, 35272, 35283, 35287, 35332, 35541, 35766, 35777, 35832, 35877, 35944, 36038, 36093, 36258, 36302, 36390, 36401, 36423, 36445, 36509, 36573, 36606, 36617, 36854, 36865, 36967, 37204, 37215, 37381, 37460, 37483, 37506, 37517, 37528, 37556, 37601, 37664, 37831, 37898, 38046, 38068, 38199, 38294, 38325, 38532, 38588, 38660, 38694, 38754, 40116, 40132, 40166, 40271, 40326, 40370, 40381, 40512, 40605, 40658, 40669, 40902, 40925, 40977, 40999, 41043, 41179, 41322, 41526, 41663, 41753, 41764, 41879, 41890, 41982, 42062, 42095, 42107, 42173, 42199, 50102, 50157, 50187, 50235, 50257, 50324, 50357, 50603, 50625, 50833, 51021, 51182, 51226, 51352, 51363, 51537, 51559, 51571, 51593, 51637, 51688, 51699, 51714, 51929, 51984, 52153, 52220, 52253, 52377, 52482, 52617, 52639, 52651, 52797, 52896, 52920, 53104, 53115, 53174, 53240, 53307, 53318, 53330, 53484, 53528, 53561, 53612, 53623, 53634, 53803, 53879, 53923, 54021, 54126, 54256, 54267, 54512, 54523, 54662, 54838, 54849, 54905, 54980, 55017, 55061, 55127, 55149, 55360, 55382, 55404, 55448, 55459, 55534, 55658, 55761, 55772, 55854, 55865, 55887, 55942, 55966, 60046, 60153, 60232, 60289, 60379, 60417, 60428, 60496, 60507, 60587, 60678, 60711, 60715, 60876, 60998, 61107, 61184, 61229, 61240, 61251, 61272, 61283, 61393, 61415, 61494, 61505, 61571, 61580, 61677, 61700, 61788, 61799, 61845, 61856, 61878, 62014, 62103, 62156, 62262, 62329, 62360, 62515, 62595, 62633, 62907, 63445, 63479, 63559, 63754, 63918, 64057, 64091, 64103, 64186, 64197, 64252, 64274, 64553, 64575, 64693, 64769, 64780, 64784, 64896, 64907, 65071, 65175, 65378, 65497, 65591, 65701, 65744, 65766, 65809, 65907, 66068, 66144, 66289, 66323, 66345, 66432, 66487, 66520, 66553, 66586, 66923, 67024, 67199, 67210, 67333, 67400, 70014, 70289, 70405, 70441, 70545, 70556, 70718, 70874, 70885, 70907, 70953, 71016, 71108, 71119, 71142, 71164, 71209, 71298, 71536, 71580, 71584, 71609, 71689, 71701, 71747, 71803, 71814, 71981, 72114, 72261, 72349, 72462, 72484, 72605, 72617, 72796, 72823, 72835, 72857, 72879, 72958, 73004, 73102, 73114, 73138, 73216, 73249, 73441, 73465, 73537, 73560, 73610, 73716, 73750, 73982, 74038, 74139, 74195, 74251, 74310, 74343, 74455, 74478, 74489, 74591, 74635, 74692, 74856, 74869, 75370, 75382, 75426, 75580, 75735, 75758, 75912, 75967, 76009, 76056, 76105, 76230, 76241, 76342, 76429, 76462, 76473, 76495, 76583, 76595, 76631, 76753, 76904, 76937, 77085, 77150, 77207, 77274, 77285, 77310, 77388, 77635, 77670, 77786, 77999, 78082, 78138, 80203, 80214, 80232, 80271, 90271, 90392, 90403, 90558, 90649, 90708, 90741, 90762, 90941, 90974, 91073, 91119, 91275, 91297, 91521, 91608, 91654, 91769, 91850, 91968, 92016, 92198, 92265, 92469, 92513, 92558, 92591, 92636, 92647, 92739, 92970, 93037, 93157, 93190, 93212, 93249, 93543, 93749, 93795, 94017, 94039, 94120, 94271, 94304, 94326, 94391, 94479, 94559, 94648, 94659, 94692, 94703, 94732, 94743, 94843, 94904, 94935, 95137, 95182, 95261, 95373, 95418, 95486, 95508, 95628, 95737, 95826, 96323, 96335, 96385, 96397, 96561, 96583, 96808, 96982, 97095, 97219, 97414, 97425, 97447, 97517, 97579, 97660, 97686, 100024, 100057, 100091, 100146, 100743, 100878, 100900, 110167, 110267, 110366, 110456, 110490, 110720, 110810, 110821, 110843, 111032, 111065, 111243, 111323, 111590, 111635, 111668, 111758, 111828, 111876, 111923, 111980, 112035, 112098, 112274, 112285, 112351, 112433, 112559, 112603, 112625, 112648, 112797, 112945, 112999, 113155, 113214, 113342, 113566, 113628, 113673, 113798, 114024, 114035, 114187, 114222, 114380, 114946, 115083, 115184, 115254, 115266, 115345, 115373, 115424, 115503, 115538, 115571, 115656, 115757, 120001, 120252, 120274, 120341, 120427, 120449, 120568, 120604, 120649, 120811, 121177, 121299, 121331, 121354, 121766, 121857, 121868, 121979, 122248, 122318, 122329, 122402, 122428, 122450, 122505, 122626, 122665, 123090, 123533, 124034, 124099, 124176, 124254, 124265, 124287, 124331, 124499, 124510, 124585, 124596, 124640, 124651, 124719, 124778, 124833, 130191, 130213, 130249, 130260, 130271, 130407, 130498, 130570, 130603, 130704, 130781, 130804, 131277, 131288, 131462, 140001, 140045, 140180, 140225, 140247, 140358, 140452, 140554, 140576, 140739, 140750, 140951, 141045, 141757, 141922, 141945]
# site_subpart_CSP_list=[134, 145, 201, 231, 253, 364, 430, 542, 634, 763, 785, 796, 963, 1262, 1358, 1539, 1569, 1602, 1638, 1663, 1729, 1861, 1884, 2121, 2143, 2165, 2199, 2403, 2432, 2506, 2815, 3053, 3719, 3798, 3809, 3977, 3999, 4517, 4670, 4813, 5026, 5060, 5138, 5257, 5323, 5484, 5528, 5832, 5854, 5921, 6148, 6227, 6239, 6377, 6388, 6501, 6535, 6587, 6610, 6643, 6655, 6835, 6977, 7021, 7458, 7661, 7839, 8100, 8410, 10051, 10229, 10284, 10516, 10727, 11167, 11299, 11310, 11664, 11793, 11960, 12136, 12214, 12493, 12526, 12558, 12730, 13651, 13989, 14135, 14257, 14590, 14940, 15232, 15640, 15652, 15675, 15708, 15882, 16057, 16090, 16438, 16449, 16472, 16543, 16981, 17175, 17275, 17651, 17696, 17707, 18505, 20127, 20193, 20344, 20703, 20803, 21034, 21113, 21147, 21591, 21807, 21851, 22302, 22515, 22737, 22847, 23101, 23167, 24109, 24391, 24535, 24546, 24759, 24781, 24814, 25094, 25681, 25726, 26393, 26404, 27022, 27760, 27826, 27904, 28153, 28379, 28693, 28748, 28895, 30156, 30299, 30561, 30712, 31035, 31068, 31090, 31478, 31583, 31687, 31832, 31854, 32166, 32233, 32394, 32427, 32934, 32956, 33125, 33281, 33519, 33531, 33656, 34082, 34160, 34929, 35096, 35443, 35911, 36049, 36115, 36225, 36324, 36628, 36672, 36945, 37023, 37228, 37261, 37732, 37743, 37853, 38120, 38232, 38283, 38798, 40036, 40070, 40105, 40144, 40199, 40315, 40414, 40447, 40521, 40565, 40823, 40988, 41032, 41076, 41356, 41435, 42312, 42334, 42512, 50080, 50168, 50176, 50213, 50290, 50439, 50550, 50614, 50672, 50719, 50963, 51341, 51429, 51440, 52269, 52355, 52598, 52663, 52863, 53207, 53409, 53495, 53591, 53704, 53766, 53868, 54245, 54444, 54501, 54883, 54930, 55248, 55327, 55580, 55805, 55827, 55842, 60012, 60484, 60954, 61207, 62549, 62617, 62835, 63363, 63468, 63570, 63628, 63639, 63665, 63778, 63800, 63844, 63851, 64586, 64625, 64940, 65186, 65264, 65462, 65508, 66045, 66312, 67109, 67177, 67257, 67268, 67312, 70130, 70366, 70729, 70929, 71198, 71264, 71478, 71632, 72025, 72384, 72550, 72561, 72729, 72784, 72890, 72993, 73261, 73295, 73396, 73662, 73694, 74085, 74106, 74117, 74602, 74745, 74834, 75547, 75592, 75603, 75945, 76572, 76970, 76992, 77252, 77377, 78093, 80048, 90034, 90111, 90341, 90504, 90594, 91702, 92041, 92052, 92302, 92380, 92547, 92681, 92704, 92992, 94062, 94176, 94871, 95094, 95283, 95307, 95672, 95803, 95874, 95929, 95951, 96873, 97326, 100467, 100585, 100630, 100809, 110101, 110134, 110145, 110730, 110775, 110797, 111991, 113043, 113203, 113376, 113499, 113577, 113763, 113856, 113879, 114079, 114141, 115173, 115746, 120034, 120139, 120471, 120681, 120767, 120845, 120951, 122010, 122036, 122058, 122171, 122193, 122516, 122784, 123374, 123665, 124187, 124353, 124554, 124789, 124855, 124904, 124949, 130237, 131019, 140067, 140437, 140794, 140870, 140962, 141022, 141076, 141378, 141412, 141512, 141534, 141559, 141779, 141812, 141860, 141997]
# # site_suball_CSP_list=[]
# # site_subpart_CSP_list=[]
#
#
# file_num=15
# for i in range(0,file_num):
#     file_name_con = file_path+"/df_csp_normfeature_sub_"+str(i)+".csv"
#     csv_data = pd.read_csv(file_name_con, low_memory=False)
#     df_csp_vec = pd.DataFrame(csv_data)
#     df_csp_vec["domain"]=df_csp_vec["domain"]+i*10000
#     csp_concat_vec_set.append(df_csp_vec)
# #######concat CSP vec
# df_csp_vec_all = pd.concat(csp_concat_vec_set,ignore_index=True,axis=0)
# df_csp_vec_all.to_csv((file_path+"/df_all_sub_features.csv"))
# df_csp_vec_all=df_csp_vec_all.drop(columns=['Unnamed: 0'])
# df_csp_vec_all["same_csp"]=0
# df_csp_vec_all["same_mode"]=0
# df_csp_vec_all["same_csp_total"]=0
# # for col in df_csp_vec_all.columns:
# #     print("col:",col)
# for index, row in df_csp_vec_all.iterrows():
#     if(row["type"]=="domain"):
#         df_csp_vec_all.at[index,"same_csp"]=-1
#         df_csp_vec_all.at[index, "same_mode"] = -1
#         df_csp_vec_all.at[index, "same_csp_total"] = -1
#     elif(row["type"]=="sub_domain"):
#         domain=row["domain"]
#         dind=df_csp_vec_all[(df_csp_vec_all["domain"]==domain) & (df_csp_vec_all["type"]=="domain")]
#         # if(len(dind)>1):
#         #    print("dind >1:",dind["Site"])
#         # if(len(dind)==0):
#         #    print("dind ==0:", dind["Site"])
#         for r_ind, d_record in dind.iterrows():
#          if(row["cspheader"].lower()==d_record["cspheader"]
#             or row["cspmetaheader"].lower()==d_record["cspmetaheader"]
#             or row["cspheader"].lower() == d_record["cspmetaheader"]
#             or row["cspmetaheader"].lower() == d_record["cspheader"]
#          ):
#             df_csp_vec_all.at[index, "same_mode"] = 1
#          same_csp=0
#          for col in df_csp_vec_all.columns:
#             if(col not in ["Site","csp_con","type","domain","cspheader","cspmetaheader","same_csp","same_mode","same_csp_total"]):
#                 if(row[col]!=d_record[col]):
#                     same_csp = same_csp+1
#          if(same_csp==0):
#             df_csp_vec_all.at[index, "same_csp"] = 1
#          if ( row["csp_con"]==d_record["csp_con"]):
#             df_csp_vec_all.at[index, "same_csp_total"] = 1
#
# df_csp_vec_all.to_csv((file_path+"/df_all_sub_features_2.csv"))
# df_or_csp=df_csp_vec_all
# print("domain set:",len(list(set(df_csp_vec_all[df_csp_vec_all["type"]=="domain"]["Site"]))))
# df_csp_vec_all=df_csp_vec_all[df_csp_vec_all["domain"].isin(site_suball_CSP_list)]
# print("site_suball_CSP_list",site_suball_CSP_list)
# group_site=df_csp_vec_all.groupby(df_csp_vec_all["domain"],as_index=False)
# print("len of group:",len(group_site))
# same_mode_all=[]
# same_csp_all=[]
# same_cspmode_all=[]
# same_cspmode_total_all=[]
# same_cspmode_all_re=[]
# same_cspmode_all_en=[]
# same_csp_total_all=[]
# same_mode_all_en=[]
# same_mode_all_re=[]
# same_cspmode_tall_re=[]
# same_cspmode_tall_en=[]
# for gname,sub_gro in group_site:
#         sumn = len(sub_gro)
#         #print("sumn:",sumn)
#         same_mode=0
#         same_csp=0
#         same_csp_total=0
#         dmain = ""
#         mode_header=""
#         mode_meta=''
#         for index, rec in sub_gro.iterrows():
#             if (rec["type"] == "domain"):
#                dmain = rec["Site"]
#                sumn=sumn-1
#                mode_header=rec["cspheader"]
#                mode_meta=rec["cspmetaheader"]
#             if(rec['same_mode']==1):
#               same_mode= 1+same_mode
#             if (rec['same_csp'] ==1):
#               same_csp=1+same_csp
#             if (rec['same_csp_total']==1): #csp content same
#               same_csp_total=1+same_csp_total
#         #print("rec:",same_mode," ",same_csp," ",same_csp_total," ",dmain)
#         if(same_mode==sumn):
#             same_mode_all.append(dmain)
#             if (mode_header.lower() == "content-security-policy" or
#                     mode_meta.lower() == "content-security-policy"):
#                 same_mode_all_en.append(dmain)
#             if (mode_header.lower()  == "content-security-policy-report-only" or
#                 mode_header.lower() == "content-security-policy-report-only"):
#                 same_mode_all_re.append(dmain)
#         if (same_csp == sumn):
#             same_csp_all.append(dmain)
#         if (same_csp_total == sumn):
#             same_csp_total_all.append(dmain)
#         if (same_csp_total == sumn and same_mode == sumn):
#             same_cspmode_total_all.append(dmain)
#             if (mode_header.lower() == "content-security-policy" or mode_meta.lower() == "content-security-policy"):
#                 same_cspmode_tall_en.append(dmain)
#             if (
#                     mode_header.lower() == "content-security-policy-report-only" or mode_meta.lower() == "content-security-policy-report-only"):
#                 same_cspmode_tall_re.append(dmain)
#         if (same_mode == sumn and same_csp == sumn):
#             same_cspmode_all.append(dmain)
#             if(mode_header.lower() =="content-security-policy" or mode_meta.lower()=="content-security-policy"):
#                 same_cspmode_all_en.append(dmain)
#             if (mode_header.lower() == "content-security-policy-report-only" or mode_meta.lower()=="content-security-policy-report-only"):
#                 same_cspmode_all_re.append(dmain)
# print("same_csp:",len(same_csp_all))
# print("same_mode:",len(same_mode_all))
# print("same_mode_csp:",len(same_cspmode_all))
# print("same_mode_csp re:",len(same_cspmode_all_re))
# print("same_mode_csp en:",len(same_cspmode_all_en))
# print("same_mode_csp total:",len(same_cspmode_total_all))
# print("same_csp total:",len(same_csp_total_all))
# print("same_mode en:",len(same_mode_all_en))
# print("same_mode re:",len(same_mode_all_re))
# print("same_mode en:",len(same_cspmode_tall_en))
# print("same_mode re:",len(same_cspmode_tall_re))
# print("#################################################################################################")
# #print("subdomain same mode set:",len(list(set(df_csp_vec_all[(df_csp_vec_all["type"]=="sub_domain") & (df_csp_vec_all["same_mode"]==1)]["domain"]))))
# #print("subdomain same csp set:",len(list(set(df_csp_vec_all[(df_csp_vec_all["type"]=="sub_domain") & (df_csp_vec_all["same_csp"]==1)]["domain"]))))
# #print("subdomain same both set:",len(list(set(df_csp_vec_all[(df_csp_vec_all["type"]=="sub_domain") & (df_csp_vec_all["same_csp"]==1) & (df_csp_vec_all["same_mode"]==1)]["domain"]))))
# df_csp_vec_all=df_or_csp[df_or_csp["domain"].isin(site_subpart_CSP_list)]
# print("site_subpart_CSP_list",site_subpart_CSP_list)
# group_site=df_csp_vec_all.groupby(df_csp_vec_all["domain"],as_index=False)
# print("len of group:",len(group_site))
# same_mode_all=[]
# same_csp_all=[]
# same_cspmode_all=[]
# same_cspmode_total_all=[]
# same_cspmode_all_re=[]
# same_cspmode_all_en=[]
# same_csp_total_all=[]
# same_mode_all_en=[]
# same_mode_all_re=[]
# same_cspmode_tall_re=[]
# same_cspmode_tall_en=[]
# for gname,sub_gro in group_site:
#         sumn = len(sub_gro)
#         #print("sumn:",sumn)
#         same_mode=0
#         same_csp=0
#         same_csp_total=0
#         dmain = ""
#         mode_header=""
#         mode_meta=''
#         for index, rec in sub_gro.iterrows():
#             if (rec["type"] == "domain"):
#                dmain = rec["Site"]
#                sumn=sumn-1
#                mode_header=rec["cspheader"]
#                mode_meta=rec["cspmetaheader"]
#             if(rec['same_mode']==1):
#               same_mode= 1+same_mode
#             if (rec['same_csp'] ==1):
#               same_csp=1+same_csp
#             if (rec['same_csp_total']==1): #csp content same
#               same_csp_total=1+same_csp_total
#         #print("rec:",same_mode," ",same_csp," ",same_csp_total," ",dmain)
#         if(same_mode==sumn):
#             same_mode_all.append(dmain)
#             if (mode_header.lower() == "content-security-policy" or
#                     mode_meta.lower() == "content-security-policy"):
#                 same_mode_all_en.append(dmain)
#             if (mode_header.lower()  == "content-security-policy-report-only" or
#                 mode_header.lower() == "content-security-policy-report-only"):
#                 same_mode_all_re.append(dmain)
#         if (same_csp == sumn):
#             same_csp_all.append(dmain)
#         if (same_csp_total == sumn):
#             same_csp_total_all.append(dmain)
#         if (same_csp_total == sumn and same_mode == sumn):
#             same_cspmode_total_all.append(dmain)
#             if (mode_header.lower() == "content-security-policy" or mode_meta.lower() == "content-security-policy"):
#                 same_cspmode_tall_en.append(dmain)
#             if (
#                     mode_header.lower() == "content-security-policy-report-only" or mode_meta.lower() == "content-security-policy-report-only"):
#                 same_cspmode_tall_re.append(dmain)
#         if (same_mode == sumn and same_csp == sumn):
#             same_cspmode_all.append(dmain)
#             if(mode_header.lower() =="content-security-policy" or mode_meta.lower()=="content-security-policy"):
#                 same_cspmode_all_en.append(dmain)
#             if (mode_header.lower() == "content-security-policy-report-only" or mode_meta.lower()=="content-security-policy-report-only"):
#                 same_cspmode_all_re.append(dmain)
# print("same_csp:",len(same_csp_all))
# print("same_mode:",len(same_mode_all))
# print("same_mode_csp:",len(same_cspmode_all))
# print("same_mode_csp re:",len(same_cspmode_all_re))
# print("same_mode_csp en:",len(same_cspmode_all_en))
# print("same_mode_csp total:",len(same_cspmode_total_all))
# print("same_csp total:",len(same_csp_total_all))
# print("same_mode en:",len(same_mode_all_en))
# print("same_mode re:",len(same_mode_all_re))
# print("same_mode en:",len(same_cspmode_tall_en))
# print("same_mode re:",len(same_cspmode_tall_re))