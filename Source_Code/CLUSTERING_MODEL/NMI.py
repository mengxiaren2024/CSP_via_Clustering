
import pandas as pd
import ssl
from sklearn.metrics.cluster import normalized_mutual_info_score

ssl._create_default_https_context = ssl._create_unverified_context

csp_dset=[]
csv_data_or = pd.read_csv("../new_or_csv_data.csv", low_memory=False)
label_or_9=csv_data_or["label_9"]
label_or_15=csv_data_or["label_15"]
label_or_5=csv_data_or["label_5"]
label_or_14=csv_data_or["label_14"]

Kmeans_256_rr_label5 = pd.read_csv("/model_test_all/256/comparson_RR_256.csv", low_memory=False)["lab_kmeans5"]
spbf_256_as_label5 = pd.read_csv("/model_test_all/256/comparson_AS_256.csv", low_memory=False)["lab_rf0_5"]
spnn_256_rs_label15=pd.read_csv("/model_test_all/256/comparson_RS_256.csv", low_memory=False)["lab_spnn_15"]


Kmeans_512_rr_label5 = pd.read_csv("/model_test_all/512/comparson_RR_512.csv", low_memory=False)["lab_kmeans5"]
spbf_512_rr_label5 =  pd.read_csv("/model_test_all/512/comparson_RR_512.csv", low_memory=False)["lab_rf0_5"]
spnn_512_rr_label9= pd.read_csv("/model_test_all/512/comparson_RR_512.csv", low_memory=False)["lab_spnn_9"]


print("NMI Kmeans_256_rr_label5:",normalized_mutual_info_score(Kmeans_256_rr_label5, label_or_5))
print("NMI spbf_256_as_label5:",normalized_mutual_info_score(spbf_256_as_label5, label_or_5))
print("NMI spnn_256_rs_label15:",normalized_mutual_info_score(spnn_256_rs_label15, label_or_15))


Kmeans_or_label5 = pd.read_csv("/model_test_all/comparson_or.csv", low_memory=False)["lab_kmeans5"]
spbf_or_label14 =  pd.read_csv("/model_test_all/comparson_or.csv", low_memory=False)["lab_rf0_14"]
spnn_or_label15= pd.read_csv("/model_test_all/comparson_or.csv", low_memory=False)["lab_spnn_15"]
print("NMI Kmeans_or_label5:",normalized_mutual_info_score(Kmeans_512_rr_label5, label_or_5))
print("NMI spbf_or_label14:",normalized_mutual_info_score(spbf_or_label14, label_or_14))
print("NMI spnn_or_label15:",normalized_mutual_info_score(spnn_or_label15, label_or_15))

