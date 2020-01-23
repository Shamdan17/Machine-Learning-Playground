#!/usr/bin/env python
# coding: utf-8
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
import csv
import numpy as np
import pandas as pd
import seaborn as sns 
import math
import matplotlib.pyplot as plt
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 

def rowToDayOfWeek(row):
    return datetime.datetime(row['YEAR'], row['MONTH'], row['DAY']).weekday()


#gets a one hot vector of weekdays for the input vector
def getDaysOfWeek(df):
    days = df.apply(rowToDayOfWeek, axis=1)
    days = pd.get_dummies(days.astype(str), prefix = "WeekDay")
    return days


def predict(X_DF, model):
    cat = ['TRX_TYPE','IDENTITY', 'REGION','MONTH','YEAR']
    days_scaled = X_DF['DAY']/31
    df_cat = X_DF[cat]
    df_cat = pd.get_dummies(df_cat.astype(str))
    X_DF = pd.concat([getDaysOfWeek(X_DF),df_cat, days_scaled], axis=1)
    cats = ['WeekDay_0','WeekDay_1','WeekDay_2','WeekDay_3','WeekDay_4','WeekDay_5','WeekDay_6','TRX_TYPE_1','TRX_TYPE_2','IDENTITY_1143004967','IDENTITY_1548999825','IDENTITY_1719546898','IDENTITY_17419935','IDENTITY_1787422269','IDENTITY_1804676971','IDENTITY_1817164026','IDENTITY_1902912455','IDENTITY_1908117039','IDENTITY_1959848986','IDENTITY_2044016829','IDENTITY_2056753503','IDENTITY_2096665437','IDENTITY_2202273924','IDENTITY_2290018626','IDENTITY_2290604134','IDENTITY_2298030412','IDENTITY_233741221','IDENTITY_2430962943','IDENTITY_2471550189','IDENTITY_2600763243','IDENTITY_2692079302','IDENTITY_2803169495','IDENTITY_280545922','IDENTITY_2845704036','IDENTITY_2923411006','IDENTITY_2953253955','IDENTITY_3335853404','IDENTITY_3427812780','IDENTITY_3429896726','IDENTITY_3584635349','IDENTITY_3601108735','IDENTITY_3630455426','IDENTITY_3664353527','IDENTITY_3838095733','IDENTITY_3879892844','IDENTITY_4023965915','IDENTITY_4029676141','IDENTITY_4238406808','IDENTITY_4258785446','IDENTITY_4264239603','IDENTITY_520462337','IDENTITY_564429665','IDENTITY_728628840','IDENTITY_746653048','IDENTITY_780319529','IDENTITY_881793234','REGION_1','REGION_10','REGION_11','REGION_12','REGION_13','REGION_14','REGION_15','REGION_16','REGION_17','REGION_18','REGION_19','REGION_2','REGION_20','REGION_3','REGION_4','REGION_5','REGION_6','REGION_7','REGION_8','REGION_9','MONTH_1','MONTH_10','MONTH_11','MONTH_12','MONTH_2','MONTH_3','MONTH_4','MONTH_5','MONTH_6','MONTH_7','MONTH_8','MONTH_9','YEAR_2018','YEAR_2019','DAY']
    X_DF = X_DF.T.reindex(cats).T.fillna(0)
    ts = torch.cuda.FloatTensor(X_DF.values)
    predictions = model(ts)
    return(predictions)


# Learning Algorithm
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(91, 75),
            nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            nn.Linear(75, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x

def main():
    # read inputs
    df = pd.read_csv("training_data.csv")
    tf = pd.read_csv("test_data.csv")

    # investigate inputs
    #sns.pairplot(df);
    df.describe()

    df.corr()

    # extract real y values
    y = df['TRX_COUNT']

    # Names of categories to split to one-hot-vectors
    cat = ['TRX_TYPE','IDENTITY', 'REGION','MONTH','YEAR']
    df_cat = df[cat]

    #Convert the types to one hot
    df_cat = pd.get_dummies(df_cat.astype(str))

    # Scale days to 0-1
    scaled_days = df['DAY']/31


    # get the updated df after the modifications
    df = pd.concat([getDaysOfWeek(df), df_cat,scaled_days], axis=1)

    # training parameters
    epoch = 15
    batchsize = 32

    # model to train
    model = MLP()
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.MSELoss()

    # copy input parameters
    x = df

    # split to training and test
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)


    # Train the model
    for trial in range(epoch):
        numBatches = math.ceil(x_train.shape[0]/batchsize)
        # for printing progress
        # cnt = 0
        for el in range(numBatches):
            optimizer.zero_grad()
            batch = x_train[el*batchsize:((el+1)*batchsize)]
            ts = torch.cuda.FloatTensor(batch.values)
            real_ys = torch.cuda.FloatTensor(np.array(y_train[el*batchsize:el*batchsize+batch.shape[0]]))

            ys = model(ts)

            loss = loss_func(real_ys, ys.squeeze())
            loss.backward()
            optimizer.step()
            # # print progress if needed
            # cnt+=1
            # if(cnt%100 == 0):
            #     print(cnt)
        
        real_ys = torch.cuda.FloatTensor(np.array(y_train))
        ys = model(torch.cuda.FloatTensor(x_train.values))

        #plot if needed
        # plt.plot(np.array(y_train), label="Train", color='Blue')
        # plt.plot(ys.cpu().detach().numpy(), label="Test", color='Red')
        # plt.show()
        pred_x = torch.cuda.FloatTensor(x_test.values)
        predYs = torch.cuda.FloatTensor(model(pred_x))
        Ys = torch.cuda.FloatTensor(y_test)
        print("training loss {}".format(loss_func(real_ys, ys.squeeze())), "test loss {}".format(loss_func(Ys,predYs.squeeze())))


    #test on training test set
    loss_func = nn.MSELoss()
    pred_x = torch.cuda.FloatTensor(x_test.values)
    predYs = torch.cuda.FloatTensor(model(pred_x))
    Ys = torch.cuda.FloatTensor(y_test)
    print(loss_func(Ys,predYs.squeeze()))


    #test on all the set
    real_ys = torch.cuda.FloatTensor(np.array(y))
    ys = model(torch.cuda.FloatTensor(df.values))
    print(loss_func(real_ys, ys.squeeze()))


    #test on the required set
    preds = predict(tf, model).squeeze()
    # print(preds)
    np.savetxt("test_predictions.csv",preds.cpu().detach().numpy(), delimiter = ',')



if __name__ == '__main__':
    main()

