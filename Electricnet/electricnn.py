import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torchsort
from fast_soft_sort_master.fast_soft_sort.pytorch_ops import soft_rank,soft_sort
import torchmetrics
import optuna
import joblib
# from lightgbm import
from sklearn.model_selection import train_test_split, ShuffleSplit, TimeSeriesSplit
import torch.nn.functional as F

class AttentionMultiHeadBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionMultiHeadBlock, self).__init__()
        self.heads = 2
        self.head_dim = output_dim // self.heads

        self.query = torch.nn.Linear(input_dim, output_dim)
        self.key = torch.nn.Linear(input_dim, output_dim)
        self.value = torch.nn.Linear(input_dim, output_dim)

        self.out = torch.nn.Linear(output_dim, output_dim)

    def forward(self, x):
        # Assuming x has shape (batch_size, input_dim)
        q = self.query(x).view(x.shape[0], self.heads, self.head_dim)
        k = self.key(x).view(x.shape[0], self.heads, self.head_dim)
        v = self.value(x).view(x.shape[0], self.heads, self.head_dim)

        attention_scores = torch.matmul(q, k.transpose(1, 2)) / self.head_dim**0.5
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended_values = torch.matmul(attention_weights, v)
        attended_values = attended_values.view(x.shape[0], -1)

        output = self.out(attended_values)
        return output


class ElectricNet(torch.nn.Module):
    def __init__(self, layers, batch_size, activation, output_dim, learning_rate, epochs=500, *args, **kwargs):
        super().__init__()  # *args, **kwargs
        self.layers = layers
        self.batch_size = batch_size
        self.activation = activation
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.best_val_loss=np.inf
        self.patience=kwargs.get('patience',50)
        self.cat_col=kwargs.get('cat_col',[])
        self.features = kwargs.get('features')
        self.num_col=[col for col in self.features if col not in self.cat_col + ['TARGET']]
        self.seed=kwargs.get('seed',0)

        self.min_max=kwargs.get('min_max',False)

        self.use_attention = kwargs.get('use_attention', False)

        self.histo_val_loss=[]
        self.histo_train_loss = []
        self.histo_epochs=[]
        self.is_plot=True

        if self.use_attention:

            self.attention_block = AttentionMultiHeadBlock(input_dim=len(self.num_col), output_dim=((len(self.num_col)+1)//2))


        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.seed)
        # print('a')
        self.hidden_dropout=kwargs.get('hidden_dropout',0)
        # self.loss_fn=torch.nn.MSELoss() # torch.nn. MSELoss
        self.model_save_path=r'/Users/jordan/Documents/project/challenge_QRT/saved_model/model_'+str(self.seed)+'.pt'
        #pearson = cos(x1 - x1.mean(dim=1, keepdim=True), x2 - x2.mean(dim=1, keepdim=True))

        list_layers = [torch.nn.Linear(layers[k], layers[k + 1]) for k in range(len(layers) - 1)]
        list_layers += [torch.nn.Linear(layers[-1], self.output_dim)]
        activation_fn = getattr(torch.nn, 'ReLU')
        architecture = []
        input_=True
        for elt in list_layers[:-1]:
            if input_:
                architecture += [elt,torch.nn.BatchNorm1d(elt.out_features),activation_fn()] #
                input_=False
            else:
                architecture += [elt,torch.nn.Dropout(p=self.hidden_dropout),activation_fn()]
        architecture += [list_layers[-1]]
        self.architecture = torch.nn.Sequential(*architecture)
        print(self.architecture)

    def loss_fn(self,target,output):
        return - torch.nn.CosineSimilarity(dim=0, eps=1e-6)(target - target.mean(dim=0, keepdim=True),
                 output - output.mean(dim=0, keepdim=True)) #+ 0.001*torch.nn.MSELoss()(target,output)
        # target_cosine = torch.ones(target.shape[0])
        # loss = torch.nn.CosineEmbeddingLoss()(input1=output, input2=target, target=target_cosine)
        # return loss

    def forward(self, x):
        # Apply attention block at the beginning
        if self.use_attention:
            x = self.attention_block(x)

        if self.min_max:
            return torch.nn.Sigmoid()(self.architecture(x))
        else:
            return self.architecture(x)

    def training_step(self, data,target):

        output = self(data)
        # loss = torch.nn.MSELoss()(target, output)
        loss = self.loss_fn(target, output)
        # loss=-torchmetrics.regression.SpearmanCorrCoef(target,output)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validation_step(self, data,target):
        with torch.no_grad():
            output = self(data)
            # val_loss = torch.nn.MSELoss()(target, output)
            # val_loss = torch.nn.CosineEmbeddingLoss()(target, output, torch.Tensor(target.size(0)).fill_(1.0))
            val_loss = self.loss_fn(target, output)
            # val_loss=-torchmetrics.regression.spearman(target,output)
        return val_loss.item()

    def train_(self, trainloader, valloader):
        # self.train()

        for epoch in range(self.epochs):
            training_loss=0
            for data,target in trainloader:
                training_loss+=self.training_step(data,target)

            training_loss=training_loss/len(trainloader)

            val_loss=self.validate(valloader)

            self.histo_epochs.append(epoch)
            self.histo_train_loss.append(training_loss)
            self.histo_val_loss.append(val_loss)

            if val_loss<self.best_val_loss:
                self.best_val_loss=val_loss
                self.early_stopping=0
                torch.save(self.state_dict(), self.model_save_path)

            else:
                self.early_stopping+=1
                if self.early_stopping>self.patience:
                    self.plot_loss()
                    print("Early stopping triggered.")
                    break
            if epoch %50==0:
                # pass
                print(f'epochs {epoch}: training_loss: {training_loss}, validation_loss:{val_loss} \t')

        self.plot_loss()
    def plot_loss(self):
        if self.is_plot:
            plt.scatter(self.histo_epochs, self.histo_train_loss, label='training_loss')
            plt.scatter(self.histo_epochs, self.histo_val_loss, label='val_loss')
            plt.title('loss evolution')
            plt.legend()
            plt.grid()
            plt.show()
            self.is_plot = False

    def validate(self,valloader):
        self.eval()
        val_loss=0
        for data, target in valloader:
            val_loss += self.validation_step(data, target)

        return val_loss/len(valloader)



    def fit(self, train_set, val_set):

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        train_set = torch.from_numpy(train_set.astype('float32').values)
        val_set = torch.from_numpy(val_set.astype('float32').values) #[self.num_col]
        train_set=torch.utils.data.TensorDataset(train_set[:,:-1],train_set[:,[-1]])
        val_set=torch.utils.data.TensorDataset(val_set[:,:-1],val_set[:,[-1]])
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size)
        valloader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size)
        self.train_(trainloader,valloader)


    def predict(self, data_test):
        self.load_state_dict(torch.load(self.model_save_path))
        pred_data = torch.from_numpy(data_test[[col for col in self.num_col if col!='TARGET']].astype('float32').values)
        return self(pred_data)






if __name__ == '__main__':
    pass
