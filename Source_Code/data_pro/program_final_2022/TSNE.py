import json
import pandas as pd
from sklearn import manifold
from sklearn import metrics
###################################################################LOGS
import numpy as np
import matplotlib.pyplot as plt

file_path="/Volumes/Elements/CSP_HOMEPAGE/result-3/"
csv_data_1 = pd.read_csv(file_path+"label_results_kmeans_elkan.csv", low_memory=False)
df_csp_vec_all2= pd.DataFrame(csv_data_1)
df_csp_vec_all=df_csp_vec_all2.drop(columns=['Unnamed: 0',"lab12"])
print("columns length:", df_csp_vec_all.columns," ***************************************")

cspvec=np.array(df_csp_vec_all)
# # ###########################################################################3d figure
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
cspv = tsne.fit_transform(cspvec)
label=np.array(df_csp_vec_all2["lab12"].to_list())
df=np.hstack((cspv,label.reshape(-1,1)))

color=["red","blue","purple","green","orange","black","brown","deeppink", "forestgreen","royalblue","navy","teal"]
la=["Cluster 1","Cluster 2","Cluster 3","Cluster 4","Cluster 5","Cluster 6","Cluster 7","Cluster 8","Cluster 9","Cluster 10","Cluster 11","Cluster 12"]






fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
u_labels = np.unique(label)
print(len(u_labels))
color=["red","blue","purple","green","orange","black","brown","deeppink", "forestgreen","yellow","navy","teal"]
for i in u_labels:
  ax.scatter(df[label == i, 0],df[label == i, 1],0, s = 40 , color = color[i], label = ("Cluster "+str(i+1)))
        #ax.scatter(cspvec[label == i, 0],cspvec[label == i, 1],cspvec[label == i, 2], s = 40 , color = color[i], label = i)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.legend()
  fil2=file_path+"Kmeans_12_elkan.png"
  #fil2 = file_path + "Kmeans_" + str(cn) + "_elkan.png"
  plt.savefig(fil2)

fig, ax = plt.subplots()
for i in u_labels:
    ax.scatter(df[label == i, 0], df[label == i, 1], s=20, color=color[i], label=("Cluster " + str(i + 1)))
ax.legend(prop={'size': 5.5}, loc='lower right')
fil2 = file_path + "Kmeans_122_elkan.png"
# fil2 = file_path + "Kmeans_" + str(cn) + "_elkan.png"
plt.savefig(fil2)