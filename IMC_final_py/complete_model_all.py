#######LENET
#https://www.researchgate.net/publication/2985446_Gradient-Based_Learning_Applied_to_Document_Recognition
#https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342
#binary representation learning
##########################input data
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.cluster import SpectralClustering
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn import metrics
from sklearn.cluster import KMeans


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
import random
import torchvision.transforms.functional as tvf
from torchvision import transforms
import numpy as np

from efficientnet_pytorch import EfficientNet
from torch.multiprocessing import cpu_count
from torch.optim import RMSprop
import pytorch_lightning as pl
color = ["red", "blue", "purple", "darkblue", "green",
         "orange", "#D12B60", "black", "brown", "deeppink",
         "olive","lawngreen", "darkorange", "forestgreen", "royalblue",
         "navy", "teal", "chocolate", "gold","yellow"]
csp_dset=[]
new_data_or = pd.read_csv("/Volumes/Elements2/CSP_SIMclr/IMC_final_py/csp_vec_final.csv", low_memory=False)
csv_data = pd.read_csv("/Volumes/Elements2/CSP_SIMclr/IMC_final_py/csp_vec_final.csv", low_memory=False)
print(csv_data.columns)
csv_data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Site', 'csp_con', 'domain'],inplace=True)
print(len(csv_data.columns))
train_dataset=[]
for index,row in csv_data.iterrows():
       train_dataset.append(np.array(row.array))
#####################################################################################################data_augementation
def data_aug_add(or_data):
    mod_data=or_data
    feature_0_index=[]
    for index in range(0,len(or_data)):
        if or_data[index]==0:
            feature_0_index.append(index)
    if len(feature_0_index) > 0:
     add_index=random.randint(0, len(feature_0_index)-1)
     mod_data[feature_0_index[add_index]]=1
    return mod_data



def data_aug_remove(or_data):
    mod_data=or_data
    feature_1_index=[]
    for index in range(0,len(or_data)):
        if or_data[index]==1:
            feature_1_index.append(index)
    if len(feature_1_index)>0:
     remove_index=random.randint(0, len(feature_1_index)-1)
     mod_data[feature_1_index[remove_index]]=0
    return mod_data

def data_aug_swap(or_data):
    mod_data=or_data
    feature_0_index=[]
    feature_1_index =[]
    for index in range(0,len(or_data)):
        if or_data[index]==1:
            feature_1_index.append(index)
        else:
            feature_0_index.append(index)

    if(len(feature_0_index)>0 and len(feature_1_index)>0):
     add_index=random.randint(0, len(feature_0_index)-1)
     remove_index = random.randint(0, len(feature_1_index) - 1)
     mod_data[feature_1_index[remove_index]]=0
     mod_data[feature_0_index[add_index]]=1
    return mod_data


def data_aug_method_selection(method,or_data):
    if method=="add":
        data_aug_add(or_data)
    if method=="remove":
        data_aug_remove(or_data)
    if method=="swap":
        data_aug_swap(or_data)

####################################################################################################set up the model




class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


##########################################################data_aug
class PretrainingDatasetWrapper(Dataset):
    def __init__(self, ds: Dataset, debug=False):
        super().__init__()
        self.ds = ds
        self.debug = debug
        if debug:
            print("DATASET IN DEBUG MODE")

        # I will be using network pre-trained on ImageNet first, which uses this normalization.
        # Remove this, if you're training from scratch or apply different transformations accordingly

    def __len__(self):
        return len(self.ds)

    def __getitem_internal__(self, idx):
        this_image_raw = self.ds[idx]
        # print("this_image_raw",this_image_raw)

        t1 = torch.tensor(data_aug_add(this_image_raw),dtype=torch.float)
        t2 = torch.tensor(data_aug_swap(this_image_raw),dtype=torch.float)

        return (t1, t2), torch.tensor(0)

    def __getitem__(self, idx):
        return self.__getitem_internal__(idx)

    def raw(self, idx):
        return self.ds[idx]

################################set up model
class ANN(nn.Module):

 def __init__(self):
  super().__init__()
  self.hidden_layer1 = nn.Linear(in_features=358, out_features=512)
  self.hidden_layer2 = nn.Linear(in_features=512, out_features=256)
  self.output_layer = nn.Linear(in_features=256, out_features=128)

 def forward(self, x):
  x = F.relu(self.hidden_layer1(x))
  x = F.relu(self.hidden_layer2(x))
  x = self.output_layer(x)
  return x

class ImageEmbedding(nn.Module):
    class Identity(nn.Module):
        def __init__(self): super().__init__()

        def forward(self, x):
            return x

    def __init__(self, embedding_size=358):
        super().__init__()
        base_model=ANN()
        internal_embedding_size =128
        self.embedding = base_model
        self.projection = nn.Sequential(
            nn.Linear(in_features=internal_embedding_size, out_features=internal_embedding_size),
            nn.ReLU(),
            nn.Linear(in_features=internal_embedding_size, out_features=internal_embedding_size)
        )

    def calculate_embedding(self, image):
        return self.embedding(image)

    def forward(self, X):
        image = X
        embedding = self.calculate_embedding(image)
        projection = self.projection(embedding)
        return embedding, projection

####################################################################################################traning process
class ImageEmbeddingModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        # print(self.hparams.lr)
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.model = ImageEmbedding()
        self.loss = ContrastiveLoss(hparams.batch_size)
        hparams_new = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams_new)

    def forward(self, X):
        return self.model(X)

    def total_steps(self):
        return len(self.train_dataloader()) // self.hparams.epochs

    def train_dataloader(self):
        return DataLoader(PretrainingDatasetWrapper(train_dataset,
                                                    debug=getattr(self.hparams, "debug", False)),
                          batch_size=self.hparams.batch_size,
                          # num_workers=cpu_count(),
                          sampler=SubsetRandomSampler(list(range(hparams["train_size"]))),
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(PretrainingDatasetWrapper(train_dataset,
                                                    debug=getattr(self.hparams, "debug", False)),
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          # num_workers=cpu_count(),
                          sampler=SequentialSampler(
                              list(range(hparams["train_size"], hparams["train_size"] + hparams["validation_size"]))),
                          drop_last=True)


    def step(self, batch, step_name="train"):
        (X, Y), y = batch
        embX, projectionX = self.forward(X)
        embY, projectionY = self.forward(Y)
        loss = self.loss(projectionX, projectionY)
        loss_key = f"{step_name}_loss"
        tensorboard_logs = {loss_key: loss}

        return {("loss" if step_name == "train" else loss_key): loss, 'log': tensorboard_logs,
                "progress_bar": {loss_key: loss}}

    def val_step(self, batch, step_name="val"):
        (X, Y), y = batch
        embX, projectionX = self.forward(X)
        embY, projectionY = self.forward(Y)
        loss = self.loss(projectionX, projectionY)
        loss_key = f"{step_name}_loss"
        tensorboard_logs = {loss_key: loss}

        return {("loss" if step_name == "train" else loss_key): loss, 'log': tensorboard_logs,
                "progress_bar": {loss_key: loss}}


    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def test_step(self, batch, batch_idx):
        return self.test_step(batch, "test")

    def validation_step(self, batch, batch_idx):
        return self.val_step(batch, "val")

    def validation_end(self, outputs):
        if len(outputs) == 0:
            return {"val_loss": torch.tensor(0)}
        else:
            loss = torch.stack([x["val_loss"] for x in outputs]).mean()
            return {"val_loss": loss, "log": {"val_loss": loss}}

    # def configure_optimizers(self):
    #     optimizer = RMSprop(self.model.parameters(), lr=self.hparams.lr)
    #     return [optimizer], []

    def configure_optimizers(self):
        optimizer = RMSprop(self.model.parameters(), lr=self.hparams.lr)
        schedulers = [
            CosineAnnealingLR(optimizer, self.hparams.epochs)
        ] if self.hparams.epochs > 1 else []
        return [optimizer], schedulers

#
from argparse import Namespace
args = { 'lr':0.0001,
    'epochs':100,
    'batch_size':193,
    'train_size':11937,
    'validation_size':1380}

#
hparams = args
module = ImageEmbeddingModule(hparams)
t = pl.Trainer(gpus=0,max_epochs=hparams['epochs'])
#
# lr_finder = t.tuner.lr_find(module)
# lr_finder.plot(show=True, suggest=True)
# print(lr_finder.suggestion())
#
#
t.fit(module)
checkpoint_file = "/Volumes/Elements2/CSP_SIMclr/IMC_final_py/simclr_AS_512.ckpt"
t.save_checkpoint(checkpoint_file)
# # # # ###################################################################################testing process+spectral_clustering
#
checkpoint_file="/Volumes/Elements2/CSP_SIMclr/IMC_final_py/simclr_AS_512.ckpt"
base_model = ImageEmbeddingModule.load_from_checkpoint(checkpoint_file).model

embeddings,projections = base_model(torch.tensor(np.array(csv_data),dtype=torch.float))

# ###################spectral clustering rfb
# print("type of embeddings:",type(embeddings))
# cspvec= embeddings.numpy()
cspvec=embeddings.detach().numpy()
####################################################SP_NN
plt.figure(figsize=(15, 15))
plot_num = 1
for cn in range(5,21):
 clf = SpectralClustering(n_clusters=cn, affinity="nearest_neighbors",n_neighbors=200)
 label=clf.fit_predict(cspvec)
 # print("eyebow score:",clf.inertia_)
 if(len(np.unique(label))>1):
    print("num:", cn , " clusters:", len(np.unique(label))," cluster eval:", metrics.silhouette_score(cspvec, label))
 for i in  np.unique(label):
    print("cluster_",i,": ", len(cspvec[label == i]))
 col = "lab_spnn_or_" + str(cn)
 new_data_or[col] = label

# # ###########################################################################3d figure
 tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
 cspv = tsne.fit_transform(cspvec)
 df=np.hstack((cspv,label.reshape(-1,1)))
 plt.subplot(4, 4, plot_num)
 u_labels = np.unique(label)
 # print(len(u_labels))
#############################################################################
 for i in u_labels:
      plt.scatter(df[label == i, 0], df[label == i, 1], s=5, color=color[i], label=("Cluster " + str(i + 1)))
 # ax.legend(prop={'size': 5.5}, loc='lower right')
 plot_num += 1

fil1="/Volumes/Elements2/CSP_SIMclr/512_AS/AS_SP_NN_512-2.png"
plt.savefig(fil1)
#
# # #######################################################Kmeans
plt.figure(figsize=(15, 15))
plot_num = 1
label_all=[]
# print("type of embeddings:",type(embeddings))
# cspvec= embeddings.detach().numpy()

for cn in range(5,21):
 clf = KMeans(n_clusters=cn,init = "k-means++")
 label=clf.fit_predict(cspvec)
 # print("eyebow score:",clf.inertia_)
 if(len(np.unique(label))>1):
    print("num:", cn , " clusters:", len(np.unique(label))," cluster eval:", metrics.silhouette_score(cspvec, label))
 for i in  np.unique(label):
    print("cluster_",i,": ", len(cspvec[label == i]))
 col = "lab_kmeans_or_" + str(cn)
 new_data_or[col] = label

# # ###########################################################################3d figure
 tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
 cspv = tsne.fit_transform(cspvec)
 df=np.hstack((cspv,label.reshape(-1,1)))
 plt.subplot(4, 4, plot_num)
 u_labels = np.unique(label)
 # print(len(u_labels))
 #color=["red","blue","purple","darkblue","green","orange","#D12B60","black","brown","deeppink","olive", "lawngreen","darkorange","forestgreen","royalblue","navy","teal","chocolate","gold"]
#############################################################################
 for i in u_labels:
      plt.scatter(df[label == i, 0], df[label == i, 1], s=5, color=color[i], label=("Cluster " + str(i + 1)))
 # ax.legend(prop={'size': 5.5}, loc='lower right')
 plot_num += 1

fil2="/Volumes/Elements2/CSP_SIMclr/512_AS/AS_kmeans_512-2.png"
plt.savefig(fil2)

new_data_or.to_csv("/Volumes/Elements2/CSP_SIMclr/512_AS/AS_512-2.csv")










