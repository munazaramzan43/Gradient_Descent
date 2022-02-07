

import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt





#Load dataset and convert it into dataframe
#(https://www.kaggle.com/jeanpierrebetancourt/houses-price-web-scraping-mexico)

df=pd.read_csv('Kaggle_house_price_Mexico_dataset.csv')
df.head(10)



#Identify missing values
df.isna().sum()


#To store changes in the df, we use argument...inplace=True and axis=1 represents the column
df.dropna(axis=1,inplace=True)



#Fill NaN values with the average value
num_data=df.select_dtypes(include=np.number)
num_cols=num_data.columns
num_cols



df[num_cols]=df[num_cols].fillna(df.mean())


#Check whether Nan values have been removed
df.isna().sum()/len(df)*100



df.head(10)



#randomly choose 5 samples from the dataset
df.sample(5)


# Data Scaling (Min-Max scaling)
from sklearn import preprocessing

Scale_X=preprocessing.MinMaxScaler()#Bring independent variables(Area,Rest_rooms,Bed_rooms) in the scale of 0&1
Scale_Y=preprocessing.MinMaxScaler()

Scaled_X= Scale_X.fit_transform(df.drop('Price',axis='columns'))
#Scaled_X
Scaled_Y= Scale_Y.fit_transform(df['Price'].values.reshape(df.shape[0],1))
Scaled_Y




Scaled_X.shape[1]


# ## Price_pred=w1*area+w2*rest_rooms+w3*bed_rooms+bias

# ### Implement Batch_Grad_desc (Simple Neural Network)



def batch_grad_desc(X,y_actual,epochs,learn_rate=0.01):
    num_feats=X.shape[1]
    
    #Find weights,and biases and initialize weights and bias randomly
    w=np.ones(num_feats)#array of weights (w1,w2,w3)
    bias=0
    samples_total=X.shape[0]
    
    cost_list=[]
    epoch_list=[]
    
    for i in range(epochs):
        y_pred= np.dot(w,Scaled_X.T)+bias  #w1*area+w2*rest_rooms+w3*bed_room+bias
        
        #Find Derivatives of error wrt weights and bias
        
        weight_grad= -(2/samples_total)*(X.T.dot(y_actual-y_pred))
        bias_grad= -(2/samples_total)*np.sum((y_actual-y_pred))
        
        w=w-learn_rate*weight_grad
        bias=bias-learn_rate*bias_grad
        
        Cost_MSE=np.mean(np.square(y_actual-y_pred))
        
        if i%10==0:
            cost_list.append(Cost_MSE)
            epoch_list.append(i)
            
    return w, bias,Cost_MSE,cost_list, epoch_list     

#Fxn. Calling
w, bias,Cost_MSE,cost_list, epoch_list=batch_grad_desc(Scaled_X,Scaled_Y.reshape(Scaled_Y.shape[0],),300)
w, bias,Cost_MSE   
 
    #w.....>w1,w2,w3 for three features
    



#plot cost/epoch
plt.xlabel("epochs")
plt.ylabel("cost_MSE")
plt.plot(epoch_list,cost_list)




#Model-prediction Rest_rooms 	Bed_rooms 	
def predict(Area,Rest_rooms,Bed_rooms,w,bias):
    
    #Trnasform fxn. input features in scale of 0&1
    Scaled_X=Scale_X.transform([[Area,Rest_rooms,Bed_rooms]])[0]
    
    #Scaled Price value(between 0&1)
    Scaled_Price=w[0]*Scaled_X[0]+w[1]*Scaled_X[1]+w[2]*Scaled_X[2]+bias   
    
    #Actual Price (Do inverse transformation of scaled price)
    return Scale_Y.inverse_transform([[Scaled_Price]])[0][0]

predict(600,4,2,w,bias)




Scale_Y.inverse_transform([[1,0]])




Scaled_Price=w[0]*Scaled_X[0]+w[1]*Scaled_X[1]+w[2]*Scaled_X[2]+bias 
Scaled_Price



Scale_X.transform([[120,1,2]])


# ### Stochastic_Grad_desc (For large datasets)


def stoc_grad_desc(X,y_actual,epochs,learn_rate=0.01):
    
    num_feats=X.shape[1]#num_features in Scaled_X (Independent features)=3
    
    w=np.ones(num_feats)#array of weights (w1,w2,w3)
    bias=0
    samples_total=X.shape[0]
    
    cost_list=[]
    epoch_list=[]
    
    for i in range(epochs):
        #we will choose random samples from our input
        rand_index=random.randint(0,samples_total-1)#index=length-1 and randint gives one integer not a range
        sample_X= X[rand_index]
        sample_Y= y_actual[rand_index]
        
        y_pred=np.dot(w,sample_X.T)+bias
        
        #Find Derivatives of error wrt weights and bias
        
        weight_grad= -(2/samples_total)*(sample_X.T.dot(sample_Y-y_pred))
        bias_grad= -(2/samples_total)*((sample_Y-y_pred))
        
        w=w-learn_rate*weight_grad
        bias=bias-learn_rate*bias_grad
        
        Cost_MSE=np.square(sample_Y-y_pred)
        
        if i%10==0:
            cost_list.append(Cost_MSE)
            epoch_list.append(i)
            
    return w, bias,Cost_MSE,cost_list, epoch_list     

#Fxn. Calling and append the values of parameters in different variables
w_stoc, bias_stoc,Cost_MSE_stoc,cost_list_stoc, epoch_list_stoc=stoc_grad_desc(Scaled_X,Scaled_Y.reshape(Scaled_Y.shape[0],),500)
w_stoc, bias_stoc,Cost_MSE_stoc  
 
    #w.....>w1,w2,w3 for three features
    
    


plt.xlabel("epoch")
plt.ylabel("cost")
plt.plot(epoch_list_stoc,cost_list_stoc)


predict(105.000000,2.000000,3.000000,w_stoc,bias_stoc)


# ## Mini_batch_grad_desc




def mini_batch_grad_desc(X,y_actual, epochs=80, batch_size=10,learn_rate=0.01):
    num_feats=X.shape[1]
    
    w=np.ones(shape=(num_feats))
    bias_m=0
    samples_total=X.shape[0]
    
    if batch_size>samples_total:
        batch_size=samples_total
        
    cost_list=[]
    epoch_list=[]
    
    num_batches=int(samples_total/batch_size)
    
    for i in range(epochs):
        rand_ind=np.random.permutation(samples_total)
        X_data_temp=X[rand_ind]
        Y_data_temp=y_actual[rand_ind]
        
        for j in range(0,samples_total,batch_size):
            X_batch=X_data_temp[j:j+batch_size]
            y_batch=Y_data_temp[j:j+batch_size]
            
            y_pred=np.dot(w,X_batch.T)+bias_m
        
        #Find Derivatives of error wrt weights and bias
        
            weight_grad= -(2/len(X_batch))*(X_batch.T.dot(y_batch-y_pred))
            bias_grad= -(2/len(X_batch))*np.sum((y_batch-y_pred))
        
            w=w-learn_rate*weight_grad
            bias_m=bias_m-learn_rate*bias_grad

            Cost_MSE=np.mean(np.square(y_batch-y_pred))

            if i%10==0:
                cost_list.append(Cost_MSE)
                epoch_list.append(i)

    return w, bias,Cost_MSE,cost_list, epoch_list     

#Fxn. Calling and append the values of parameters in different variables
w_mini, bias_mini,Cost_MSE_mini,cost_list_mini, epoch_list_mini=mini_batch_grad_desc(Scaled_X,Scaled_Y.reshape(Scaled_Y.shape[0],),epochs=80,batch_size=5)
w_mini, bias_mini,Cost_MSE_mini  
 
    #w.....>w1,w2,w3 for three features
    
   

plt.xlabel("epochs")
plt.ylabel("cost")
plt.plot(epoch_list_mini,cost_list_mini)

