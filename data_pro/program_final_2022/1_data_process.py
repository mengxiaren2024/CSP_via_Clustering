import json
import pandas as pd
from functools import reduce


file_path="/Volumes/Elements/NEW_CSP_DATA/"
Datafolder=["TOP-1000"]
data_numb=1
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
    df_rl_1=df_rl_1[df_rl_1["type"]=="domain"]
    df_rl_1.to_csv((file_name+"/df_srl.csv"))
    ######################################################################################################
    file_name_stc_1 = file_name+'/Status_Code_1.csv'
    csv_data_stc = pd.read_csv(file_name_stc_1, low_memory=False)  # 防止弹出警告
    df_stc_1 = pd.DataFrame(csv_data_stc)
    df_stc_1.rename(columns={'siteurl': 'site_url'}, inplace=True)
    print("df_stc_1 columns:",df_stc_1.columns)
    # remove duplicates
    df_stc_1.drop_duplicates(['id', 'site_url'], 'last', inplace=True)
    df_stc_1 = df_stc_1[((df_stc_1['StatusC'] > 200) | (df_stc_1['StatusC'] == 200)) & (df_stc_1['StatusC'] < 300)]
    df_stc_1.to_csv((file_name + "/df_stc.csv"))
    ######################################################################################################
    file_name_logs_1 = file_name+'/logs_nov_'+str(data_numb)+'.json'
    js_logs_1 = []
    with open(file_name_logs_1, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            js_logs_1.append(json.loads(line))
    f.close()
    df_logs_1 = pd.DataFrame(js_logs_1)
    print("df_logs_1 columns:",df_logs_1.columns)
    df_logs_1.to_csv((file_name+"/df_logs_1.csv"))
    # remove duplicates
    group_logs = df_logs_1.groupby('DocumentUri')
    start_time = group_logs["Site_num"].first();
    gro_num = 0;
    gro_len = 0;
    print("group:", len(group_logs))
    for name, group in group_logs:
        for index, row in group.iterrows():
            if (float(row["Site_num"]) - float(start_time[gro_num]) > 60000):
                # print("del")
                group.drop(index=index, inplace=True)
        gro_num = gro_num + 1
        gro_len = gro_len + len(group)
    print("group:", gro_len)
    ####################################################################################################
    file_name_csph_1 = file_name+'/csp_header_nov_'+str(data_numb)+'.json'
    js_csph_1 = []
    with open(file_name_csph_1, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            js_csph_1.append(json.loads(line))
    f.close()
    df_csph_1 = pd.DataFrame(js_csph_1)
    df_csph_1["site_url"] = ''
    # add one more column for concating with logs
    for index, row in df_csph_1.iterrows():
        df_csph_1.at[index, "site_url"] = row["siteurl"].encode("utf-8").decode()
    #df_csph_1.to_csv(file_name+'/df_csph.csv')
    df_csph_1.rename(columns={'siteurl': 'DocumentUri'}, inplace=True)
    print(df_csph_1.columns)
    # remove duplicates
    df_csph_1.drop_duplicates(['id', 'DocumentUri', 'cspheader', 'cspcontent'], 'first', inplace=True)
    df_csph_1.to_csv((file_name+"/df_csph_1.csv"))
    ######################################################################################################
    file_name_meta_1 = file_name+'/CSP_Meta_1.csv'
    csv_data_meta = pd.read_csv(file_name_meta_1, low_memory=False)  # 防止弹出警告
    df_meta_1 = pd.DataFrame(csv_data_meta)
    print("meta_1", len(df_meta_1))
    print(df_meta_1.columns)
    # remove duplicates
    df_meta_1.drop_duplicates(['site_url', 'cspmetaheader', "cspmetacon"], 'last', inplace=True)
    df_meta_1.to_csv((file_name + "/df_meta_1.csv"))
    ######################################################################################################
    file_name_req_1 = file_name+'/Request_1.csv'
    csv_data_req = pd.read_csv(file_name_req_1, low_memory=False)  # 防止弹出警告
    df_req_1 = pd.DataFrame(csv_data_req)
    print("req_1", len(df_req_1))
    print(df_req_1.columns)
    # filter then remove dumplicates
    # remove duplicates
    df_req_1.drop_duplicates(['site', 'reqSite', 'status',
           'contentType', 'method'], 'last',inplace=True)
    df_req_1.to_csv((file_name+"/df_req_1.csv"))
    ####################################################################################
    file_name_ht_1 = file_name+"/Html_1.csv"
    csv_data = pd.read_csv(file_name_ht_1, low_memory=False)
    df_ht_1 = pd.DataFrame(csv_data)
    df_ht_1.drop_duplicates(["site_url"], 'first', inplace=True)
    print("html_1:", df_ht_1.columns)
    df_ht_1.to_csv((file_name+"/df_ht_1.csv"))
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
    df_concat.to_csv((file_name+'/df_concat.csv'))

    dfs_2 = [df_concat, df_rl_1]
    df_concat_2 = reduce(lambda left, right: pd.merge(left, right, on='short_url'), dfs_2)
    print("final_2", len(df_concat_2))

    delete_domain = []
    total_sit = df_concat_2.groupby(df_concat_2['domain'], as_index=False)
    for df_sub in total_sit:
        if ("domain" not in df_sub[1]["type"].values):#filter subpage without domain pages
            delete_domain.append(df_sub[1]["domain"].iloc[0])
    print("delete domain:",delete_domain)

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
    df_concat_2.to_csv((file_name+"/df_concat_2.csv"))
    df_concat_3= df_concat_2.drop(columns=["dom"])
    df_concat_3.to_csv(("/Volumes/Elements/NEW_CSP_DATA/Whole_data/df_concat_c"+str(data_numb)+".csv"))


    dfs_3= [df_concat_2,df_logs_1]
    df_concat_log = reduce(lambda left,right: pd.merge(left,right,on='DocumentUri'), dfs_3)
    df_concat_log.to_csv((file_name+"/df_concat_log.csv"))

    dfs_4 = [df_concat_2, df_req_1]
    df_req_1.rename(columns={'site': 'site_url_y'}, inplace=True)
    df_concat_req = reduce(lambda left, right: pd.merge(left, right, on='site_url_y'), dfs_4)
    df_concat_req.to_csv((file_name+"/df_concat_req.csv"))
    data_numb=data_numb+1


