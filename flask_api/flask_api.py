from flask import Flask, render_template,request
    # pytorch mlp for multiclass classification
from numpy import vstack
from numpy import argmax
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.serialization import load
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD, optimizer
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import torch


app = Flask(__name__)


@app.route('/',methods=['GET', 'POST'])
def index():
    if request.method=='POST':
        sepal_length=float(request.form['sl'])
        sepal_width=float(request.form['sw'])
        petal_length=float(request.form['pl'])
        petal_width=float(request.form['pw'])

    else:
        return render_template('index.html')

# dataset definition
    class CSVDataset(Dataset):
        # load the dataset
        def __init__(self, path):
            # load the csv file as a dataframe
            df = read_csv(path, header=None)
            global  ls
            ls=df[4].unique()
            print(ls)    
            # store the inputs and outputs
            self.X = df.values[:, :-1]
            self.y = df.values[:, -1]
            # ensure input data is floats
            self.X = self.X.astype('float32')
            # label encode target and ensure the values are floats
            self.y = LabelEncoder().fit_transform(self.y)

        # number of rows in the dataset
        def __len__(self):
            return len(self.X)

        # get a row at an index
        def __getitem__(self, idx):
            return [self.X[idx], self.y[idx]]

        # get indexes for train and test rows
        def get_splits(self, n_test=0.33):
            # determine sizes
            test_size = round(n_test * len(self.X))
            train_size = len(self.X) - test_size
            # calculate the split
            return random_split(self, [train_size, test_size])

    # model definition
    class MLP(Module):
        # define model elements
        def __init__(self, n_inputs):
            super(MLP, self).__init__()
            # input to first hidden layer
            self.hidden1 = Linear(n_inputs, 10)
            kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
            self.act1 = ReLU()
            # second hidden layer
            self.hidden2 = Linear(10, 8)
            kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
            self.act2 = ReLU()
            # third hidden layer and output
            self.hidden3 = Linear(8, 3)
            xavier_uniform_(self.hidden3.weight)
            self.act3 = Softmax(dim=1)

        # forward propagate input
        def forward(self, X):
            # input to first hidden layer
            X = self.hidden1(X)
            X = self.act1(X)
            # second hidden layer
            X = self.hidden2(X)
            X = self.act2(X)
            # output layer
            X = self.hidden3(X)
            X = self.act3(X)
            return X

    # prepare the dataset
    def prepare_data(path):
        # load the dataset
        dataset = CSVDataset(path)
        # calculate split
        train, test = dataset.get_splits()
        # prepare data loaders
        train_dl = DataLoader(train, batch_size=32, shuffle=True)
        test_dl = DataLoader(test, batch_size=1024, shuffle=False)
        return train_dl, test_dl

    # train the model
    def train_model(train_dl, model):
        # define the optimization
        criterion = CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        # enumerate epochs
        for epoch in range(500):
            # enumerate mini batches
            for i, (inputs, targets) in enumerate(train_dl):
                # clear the gradients
                optimizer.zero_grad()
                # compute the model output
                yhat = model(inputs)
                # calculate loss
                loss = criterion(yhat, targets)
                # credit assignment
                loss.backward()
                # update model weights
                optimizer.step()

    # evaluate the model
    def evaluate_model(test_dl, model):
        predictions, actuals = list(), list()
        for i, (inputs, targets) in enumerate(test_dl):
            # evaluate the model on the test set
            yhat = model(inputs)
            # retrieve numpy array
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
            # convert to class labels
            yhat = argmax(yhat, axis=1)
            # reshape for stacking
            actual = actual.reshape((len(actual), 1))
            yhat = yhat.reshape((len(yhat), 1))
            # store
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = vstack(predictions), vstack(actuals)
        # calculate accuracy
        acc = accuracy_score(actuals, predictions)
        return acc

    # make a class prediction for one row of data
    def predict(row, model):
        # convert row to data
        row = Tensor([row])
        # make prediction
        yhat = model(row)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        return yhat

    # prepare the data
    path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
    train_dl, test_dl = prepare_data(path)
    # print(len(train_dl.dataset), len(test_dl.dataset))
    # define the network
    model = MLP(4)
    #saving the model
    # checkpoint = {'state_dict': model.state_dict()}
    # torch.save(checkpoint, 'Checkpoint.pth')
    
    # train the model
    train_model(train_dl, model)
    # evaluate the model
    acc = evaluate_model(test_dl, model)
    print('Accuracy: %.3f' % acc)
    # make a single prediction
    row = [sepal_length,sepal_width,petal_length,petal_width]
    yhat = predict(row, model)
    print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
    return render_template('index.html',result='Accuracy: %.3f' % acc,predicted=ls[int(argmax(yhat))])

    # return render_template('index.html',result='Accuracy: %.3f' % acc,predicted='Predicted: %s (class=%d)' % (yhat, argmax(yhat)))

@app.route('/json_data',methods=['GET','POST'])
def jyt():
    # content=request.get_json(silent=True)
    # dt=content
    # print(dt)
    row=[]
    dt={'sepalLength': 5.1, 'sepalWidth': 3.5, 'petalLength': 1.4, 'petalWidth': 0.2}
    for v in dt.values():
        row.append(v)

    print(row)
    model=torch.load('iris-pytorch.pkl')
    # print(type(model))

    # predict function
    def predict(row, model):
        # convert row to data
        row = Tensor([row])
        # make prediction
        yhat = model(row)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        return yhat

    yhat = predict(row, model)
    # print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
    # return render_template('index.html',predicted=ls[int(argmax(yhat))])

    # yhat = predict(row, model)
    return {"row":row}




if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8000, debug=True)



