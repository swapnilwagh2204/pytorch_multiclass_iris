import pickle
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from torch.functional import Tensor
import torch.nn.functional as F

from flask import Flask, render_template,request
app = Flask(__name__)

class Model(nn.Module):
        def __init__(self, input_dim):
            super(Model, self).__init__()
            self.layer1 = nn.Linear(input_dim,50)
            self.layer2 = nn.Linear(50, 20)
            self.layer3 = nn.Linear(20, 3)
            
        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            x = F.softmax(self.layer3(x)) # To check with the loss function
            return x
'''
features, labels = load_iris(return_X_y=True)
features_train,features_test, labels_train, labels_test = train_test_split(features, labels, random_state=42, shuffle=True)

Training
model = Model(features_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
epochs = 100
x_train, y_train = Variable(torch.from_numpy(features_train)).float(), Variable(torch.from_numpy(labels_train)).long()
for epoch in range(1, epochs+1):
    # print ("Epoch #",epoch)
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    # print_(loss.item())
    
    # Zero gradients
    optimizer.zero_grad()
    loss.backward() # Gradients
    optimizer.step() # Update

predictions
x_test = Variable(torch.from_numpy(features_test)).float()
pred = model(x_test)
pred = pred.detach().numpy()
    
saving the model 
torch.save(model, "swap.pkl")
'''
@app.route('/',methods=['GET', 'POST'])
def index():
    if request.get_json() is not None:
        content=request.get_json(silent=True)
        sepal_length =content['sepal_length']
        sepal_width =content['sepal_width']
        petal_length =content['petal_length']
        petal_width =content['petal_width']

    elif request.method=='POST':    
        sepal_length=float(request.form['sl'])
        sepal_width=float(request.form['sw'])
        petal_length=float(request.form['pl'])
        petal_width=float(request.form['pw'])

    else:
        return render_template('index.html')

    saved_model = torch.load("swap.pkl")
    row=Tensor([sepal_length,sepal_width,petal_length,petal_width])
    n=np.argmax(saved_model(row).detach().numpy(), axis=0)
    columns=['Setosa','Versicolour','Virginica']
    return {"class":columns[n]}
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)

