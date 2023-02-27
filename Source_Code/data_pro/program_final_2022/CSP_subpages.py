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

file_path="/Volumes/Elements/CSPNEW_DATA/"
#Datafolder=["CSPNEW01","CSPNEW02","CSPNEW03","CSPNEW04","CSPNEW05-1","CSPNEW05-2","CSPNEW07-1","CSPNEW07-2","CSPNEW08-1","CSPNEW08-2","CSPNEW09-1","CSPNEW10-1","CSPNEW09-2","CSPNEW10-2"]
Datafolder=["CSPNEW06"]
for dafolder in Datafolder:
    file_name=file_path+dafolder
    file_name_rl_1 = file_name+"/S_RL_1.csv"
    csv_data = pd.read_csv(file_name_rl_1, low_memory=False)
    df_rl_1 = pd.DataFrame(csv_data)
    print("s_rl_1:", len(df_rl_1))
    df_rl_1["short_url"] = ""
    for index, row in df_rl_1.iterrows():
        x = row["site_url"];
        if (row["site_url"].split("/")[0] == "http:"):
            x = row["site_url"].replace("http://", "https://");
        x = x.replace("/?", "?");
        if (x.find("#")):
            x = x.split("#")[0];
        if (x.find("@")):
            x = x.split("@")[0];
        if (x[len(x) - 1] == "/"):
            x = x[0:len(x) - 1];
        df_rl_1.at[index, "short_url"] = x;
    print("columns of s_rl:",df_rl_1.columns)
    # remove duplicates
    df_rl_1.drop_duplicates(['id', 'site_url', 'type', 'domain'], 'first', inplace=True)
    df_rl_1.to_csv((file_name+"/df_srl_2.csv"))
    ######################################################################################################
    file_name_stc_1 = file_name+"/df_stc.csv"
    csv_data_stc = pd.read_csv(file_name_stc_1, low_memory=False)  # 防止弹出警告
    df_stc_1 = pd.DataFrame(csv_data_stc)
    ######################################################################################################
    file_name_log_1 = file_name + "/df_logs_1.csv"
    csv_data_log = pd.read_csv(file_name_log_1, low_memory=False)  # 防止弹出警告
    df_logs_1 = pd.DataFrame(csv_data_log)
    print("df_logs_1 columns:",df_logs_1.columns)
    ####################################################################################################
    file_name_csph_1 = file_name + "/df_csph_1.csv"
    csv_data_csph = pd.read_csv(file_name_csph_1, low_memory=False)  # 防止弹出警告
    df_csph_1 = pd.DataFrame(csv_data_csph)
    ######################################################################################################
    file_name_meta_1 = file_name+"/df_meta_1.csv"
    csv_data_meta = pd.read_csv(file_name_meta_1, low_memory=False)  # 防止弹出警告
    df_meta_1 = pd.DataFrame(csv_data_meta)
    print("meta_1", len(df_meta_1))
    print(df_meta_1.columns)

    ######################################################################################################
    file_name_req_1 = file_name+"/df_req_1.csv"
    csv_data_req = pd.read_csv(file_name_req_1, low_memory=False)  # 防止弹出警告
    df_req_1 = pd.DataFrame(csv_data_req)
    print("req_1", len(df_req_1))
    print(df_req_1.columns)
    ####################################################################################
    file_name_ht_1 = file_name+"/df_ht_1.csv"
    csv_data = pd.read_csv(file_name_ht_1, low_memory=False)
    df_ht_1 = pd.DataFrame(csv_data)
    ##############################################################################count_num for different tag
    dfs = [df_meta_1, df_csph_1, df_stc_1, df_ht_1]
    df_concat = reduce(lambda left, right: pd.merge(left, right, on='site_url'), dfs)
    print("final", len(df_concat))
    df_concat.drop_duplicates(['site_url', 'cspheader', 'cspmetaheader', 'cspcontent', "cspmetacon"], 'last',
                              inplace=True)
    print("final", len(df_concat))
    df_concat["short_url"] = ""
    for index, row in df_concat.iterrows():
        x = row["site_url"];
        if (row["site_url"].split("/")[0] == "http:"):
            x = row["site_url"].replace("http://", "https://");
        x = x.replace("/?", "?");
        if (x.find("#")):
            x = x.split("#")[0];
        if (x.find("@")):
            x = x.split("@")[0];
        if (x[len(x) - 1] == "/"):
            x = x[0:len(x) - 1];
        df_concat.at[index, "short_url"] = x;
    df_concat.to_csv((file_name+'/df_concat_sub.csv'))

    dfs_2 = [df_concat, df_rl_1]
    df_concat_2 = reduce(lambda left, right: pd.merge(left, right, on='short_url'), dfs_2)
    print("final_2", len(df_concat_2))

    delete_domain = []
    total_sit = df_concat_2.groupby(df_concat_2['domain'], as_index=False)
    for df_sub in total_sit:
        if ("domain" not in df_sub[1]["type"].values):
            delete_domain.append(df_sub[1]["domain"].iloc[0])
    print(delete_domain)

    df_concat_2 = df_concat_2.drop(df_concat_2[(df_concat_2["domain"].isin(delete_domain))].index)
    df_concat_2.drop_duplicates(['site_url_x'], 'last', inplace=True)
    print("final collected valid websites:", len(df_concat_2.groupby(df_concat_2['domain'], as_index=False)))
    print("final collected valid webpages:", len(df_concat_2.groupby(df_concat_2['DocumentUri'], as_index=False)))
    df_concat_2["ctype"] = ''
    for index, row in df_concat_2.iterrows():
        if ((row['cspheader'] == 'designed_CSP_only') & (row['cspmetaheader'] == 'none_csp')):
            df_concat_2.at[index, "ctype"] = "NCSP"
        else:
            df_concat_2.at[index, "ctype"] = "CSP"
    #df_concat_2=df_concat_2[df_concat_2["type"]=="domain"]
    df_concat_2.to_csv((file_name+"/df_concat_sub_2.csv"))