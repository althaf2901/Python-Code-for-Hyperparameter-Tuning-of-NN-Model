get_ipython().system('pip install scikit-optimize')
get_ipython().system('pip install tensorflow')
import pandas as pd
import numpy as np
df_train= pd.read_csv('C://Users//CCES-14//Desktop//alm//X//axial//bs//X_B_axial_Training.csv')
df_train = df_train[['alpha', 'beta', 'gamma', 'tau','scf_bs']]
df_val = pd.read_csv('C://Users//CCES-14//Desktop//alm//X//axial//bs//X_B_axial_Validation.csv')
df_val = df_val[['alpha', 'beta', 'gamma', 'tau','scf_bs']]
df_test = pd.read_csv('C://Users//CCES-14//Desktop//alm//X//axial//bs//X_B_axial_Testing.csv')
df_test = df_test[['alpha', 'beta', 'gamma', 'tau','scf_bs']]
#data preprocessing
#if output is NaN
nan_index = []
for i in range(len(df_train['scf_bs'])):
  if np.isnan(df_train['scf_bs'][i])==True:
    nan_index.append(i)
df_train = df_train.drop(index = nan_index)
df_train = df_train.reset_index(drop=True)
#scf_cc<0.4
scf_index = []
for i in range(len(df_train['scf_bs'])):
  if df_train['scf_bs'][i]<0:
    scf_index.append(i)
df_train = df_train.drop(index = scf_index)
df_train = df_train.reset_index(drop=True)
df_train
#data preprocessing
#if output is NaN
nan_index = []
for i in range(len(df_val['scf_bs'])):
  if np.isnan(df_val['scf_bs'][i])==True:
    nan_index.append(i)
df_val = df_val.drop(index = nan_index)
df_val = df_val.reset_index(drop=True)
#scf_cc<0.4
scf_index = []
for i in range(len(df_val['scf_bs'])):
  if df_val['scf_bs'][i]<0:
    scf_index.append(i)
df_val = df_val.drop(index = scf_index)
df_val = df_val.reset_index(drop=True)
nan_index = []
for i in range(len(df_test['scf_bs'])):
  if np.isnan(df_test['scf_bs'][i])==True:
    nan_index.append(i)
df_test = df_test.drop(index = nan_index)
df_test = df_test.reset_index(drop=True)
#scf_cc<0.4
scf_index = []
for i in range(len(df_test['scf_bs'])):
  if df_test['scf_bs'][i]<0:
    scf_index.append(i)
df_test = df_test.drop(index = scf_index)
df_test = df_test.reset_index(drop=True)
#merging both the pandas dataframes
# Stack the DataFrames on top of each other
df1 = pd.concat([df_train, df_val,df_test], ignore_index=True)
dfi = pd.DataFrame()
dfi = dfi.append(df1.iloc[:, 0:4])
dfo = pd.DataFrame()
dfo = dfo.append(df1.iloc[:, 4:5])
#different styles of hidden layers 
#rectangle
from sklearn.preprocessing import MinMaxScaler
# scale = StandardScaler()
scale = MinMaxScaler()
scaleddfi = scale.fit_transform(dfi)
scaleddfo = scale.fit_transform(dfo)
#dividing into training and testing for scf_cc
#loading the dataset
from sklearn.model_selection import train_test_split
X_train = scaleddfi[0:len(df_train), :] #.to_numpy()
y_train = scaleddfo[:,0][0:len(df_train)].reshape(len(df_train), 1) #for scf_bs
# y = scaleddfo[:,0].reshape(X.shape[0], 1)#.to_numpy() #transpose
X_val = scaleddfi[len(df_train):len(df_train)+len(df_val), :]
y_val = scaleddfo[:,0][len(df_train):len(df_train)+len(df_val)].reshape(len(df_val), 1)
X_test = scaleddfi[len(df_train)+len(df_val):, :]
y_test = scaleddfo[:,0][len(df_train)+len(df_val):].reshape(len(df_test), 1)
y_test
len(y_train)
#for 500 datapoints
import tensorflow as tf
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.sampler import Sobol
from tensorflow.keras import backend as K
import pandas as pd
# Define the search space for the hyperparameters
search_space = [Real(0.001, 0.1, name='learning_rate'),
                Integer(1, 100, name='batch_size'),
                Integer(2, 10, name='num_layers'),
                Integer(10, 200, name='num_neurons'),
                Categorical(['relu', 'tanh', 'sigmoid'], name='activation'),
                Real(0, 0.5, name='dropout')]
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
# Initialize the Sobol sampler
sampler = Sobol()
# Initialize the optimizer with the search space and sampler
optimizer = Optimizer(dimensions=search_space, random_state=123, base_estimator="GP", acq_func="EI", acq_optimizer="lbfgs")
# Define the TensorFlow neural network model
def create_model(params):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(params['num_neurons'],kernel_initializer='normal', activation=params['activation'], input_shape=(4,)))
    for i in range(params['num_layers']-1):
        model.add(tf.keras.layers.Dense(params['num_neurons']/(2**(i)),kernel_initializer='normal', activation=params['activation']))
        model.add(tf.keras.layers.Dropout(params['dropout']))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']), loss=rmse, metrics=[rmse])
    return model
# Define the objective function to optimize
@use_named_args(search_space)
def objective(**params):
    model = create_model(params)
    history = model.fit(X_train, y_train, batch_size=params['batch_size'], epochs= 1000, validation_data=(X_val, y_val), verbose=1)
    history_df = pd.DataFrame(model.history.history)
    score = history_df['val_loss'].min()
    return score
# Generate the initial points using Sobol sequence
n_points = 500
initial_points = sampler.generate(search_space, n_points)
# Create a list to store the trial points, optimization points, and best hyperparameters
points = []
# Evaluate the initial points and tell the optimizer the corresponding function values
for point in initial_points:
    value = objective(point)
    optimizer.tell(point, value)
    points.append([point, value, 'trial'])
# Continue the optimization process
n_iterations = 100
for i in range(n_iterations):
    next_point = optimizer.ask()
    print(next_point)
    value = objective(next_point)
    optimizer.tell(next_point, value)
    points.append([next_point, value, 'optimization'])
columns  = ['learning_rate', 'batch_size', 'num_layers','num_neurons', 'activation','dropout', 'val_loss', 'point_type']
# Create an empty DataFrame with the desired column names
df = pd.DataFrame()
temp_df1 = pd.DataFrame()
# temp_df1 = pd.DataFrame(columns= ['learning_rate', 'batch_size', 'num_neurons', 'activation','dropout'])
temp_df2 = pd.DataFrame()
# Iterate through the list and append each row to the DataFrame
for row in points:
    # Extract the nested list in the first column and create a new DataFrame with it
    temp_df1 = temp_df1.append([row[0]])
for row in points:
    temp_df2 = temp_df2.append([row[1:]])
df = pd.concat([temp_df1, temp_df2] , axis=1)  
# # Rename the columns of the main DataFrame
df.columns = columns
# # Reorder the columns as desired
df = df[columns]
# reset the index
df = df.reset_index()
df.loc[df['val_loss'].idxmin(),'point_type'] = 'best'
df.to_csv('C://Users//CCES-14//Desktop//alm//X//Axial//bs//X_axial_bs_dp')
best_params = df.loc[df['val_loss'].idxmin()]
best_value = best_params['val_loss']
print("Best hyperparameters: ", best_params)
print("Corresponding validation loss: ", best_value)
# Save the history of the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(best_params['num_neurons'], activation=best_params['activation'],kernel_initializer='normal', input_shape=(4,)))
for i in range(best_params['num_layers']-1):
  model.add(tf.keras.layers.Dense(best_params['num_neurons']/(2**(i)), kernel_initializer='normal', activation=best_params['activation']))
  model.add(tf.keras.layers.Dropout(best_params['dropout']))
model.add(tf.keras.layers.Dense(1, activation='linear'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']), loss=rmse, metrics=[rmse])
#saving the plots 
from keras.callbacks import ModelCheckpoint
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}-X_axial-bs.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]
res = model.fit(X_train, y_train, epochs=1000, batch_size=best_params['batch_size'], validation_data = (X_val, y_val),  callbacks=callbacks_list, verbose = 1)
history_df = pd.DataFrame(model.history.history)
history_df[['loss', 'val_loss']].plot(xlabel="Epoch", ylabel="Loss")
# model.save('/content/drive/MyDrive/results_MTP/model_inplane_bc.h5')
min_loss = history_df[['val_loss']].min()
print(min_loss)
print("Evaluate model on train data")
results = model.evaluate(X_train, y_train, batch_size=best_params['batch_size'])
print("test loss, test acc:", results)
print("Evaluate model on val data")
results = model.evaluate(X_val, y_val, batch_size=best_params['batch_size'])
print("test loss, test acc:", results)
print("Evaluate model on test data")
results = model.evaluate(X_test, y_test, batch_size=best_params['batch_size'])
print("test loss, test acc:", results)
import tensorflow as tf
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.sampler import Sobol
from tensorflow.keras import backend as K
import pandas as pd
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
df = pd.read_csv('C://Users//CCES-14//Desktop//alm//X//axial//bs//X_axial_bs_dp')
df.to_csv('X_axial_bs_dp.csv')
best_params = df.loc[df['val_loss'].idxmin()]
best_value = best_params['val_loss']
print("Best hyperparameters: ", best_params)
print("Corresponding validation loss: ", best_value)
# Save the history of the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(best_params['num_neurons'], activation=best_params['activation'],kernel_initializer='normal', input_shape=(4,)))
for i in range(best_params['num_layers']-1):
  model.add(tf.keras.layers.Dense(best_params['num_neurons']/(2**(i)), kernel_initializer='normal', activation=best_params['activation']))
  model.add(tf.keras.layers.Dropout(best_params['dropout']))
model.add(tf.keras.layers.Dense(1, activation='linear'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']), loss=rmse, metrics=[rmse])
#saving the plots 
from keras.callbacks import ModelCheckpoint
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}-X_axial-bs.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]
res = model.fit(X_train, y_train, epochs=1000, batch_size=best_params['batch_size'], validation_data = (X_val, y_val),  callbacks=callbacks_list, verbose = 1)
history_df = pd.DataFrame(model.history.history)
history_df[['loss', 'val_loss']].plot(xlabel="Epoch", ylabel="Loss")
# model.save('/content/drive/MyDrive/results_MTP/model_inplane_bc.h5')
history_df.to_csv('C://Users//CCES-14//Desktop//alm//X//axial//bs//X_axial_bs_history_df.csv')
min_loss = history_df[['val_loss']].min()
print(min_loss)
print("Evaluate model on test data")
#results = model.evaluate(X_test_all, y_test_all, batch_size=best_params['batch_size'])
print("test loss, test acc:", results)
#vertically stack arrays
X = np.vstack((X_train, X_val, X_test))
X
#vertically stack arrays
y = np.vstack((y_train, y_val, y_test))
y
## training dataset
import numpy as np
# Load wights file of the best model :
weights_file = 'Weights-819--0.01189-X_axial-bs.hdf5' # choose the best checkpoint 
model.load_weights(weights_file) # load it
# model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
# print("Evaluate model on test data")
# results = model.evaluate(X_test, y_test, batch_size=128)
# print("test loss, test acc:", results)
#computing r2_score for all the actual and predicted output
y_predict_all =  model.predict(X)
y_predict_rescale_all = []
min = dfo['scf_bs'].min() #for scf_bs
max = dfo['scf_bs'].max() #for scf_bs
for i in range(0,y_predict_all.size,1):
  y_predict_rescale_all.append((y_predict_all[i]*(max-min))+min)
y_predict_rescale_all=np.array(y_predict_rescale_all)
req = np.array(df1['scf_bs']) #for scf_bs
req=req[:].reshape(req.size, 1)
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [7.50, 7.50]
plt.rcParams["figure.autolayout"] = True
# p = y[:,0].reshape(y.size, 1)
p = req
q = y_predict_rescale_all[:,0].reshape(y_predict_all.size, 1)
# 100 linearly spaced numbers
x = np.linspace(0,max,100)
# the function, which is y = x^2 here
y = x
plt.title("After rescaling")
plt.scatter(p, q, color="green")
# plot the function
plt.plot(x,y, 'r')
plt.xlabel('y_true')
plt.ylabel('y_predicted')
plt.show()
from sklearn.metrics import r2_score
r2 = r2_score( y_predict_rescale_all[:,0].reshape(y_predict_all.size, 1), req)
print('r2 score for perfect model is', r2)
import pandas as pd
import numpy as np
dataset = pd.DataFrame({'Predicted Value': list(q), 'Actual Value': list(p)}, columns=['Predicted Value', 'Actual Value'])
# Define a function to remove brackets and convert to float
def remove_brackets_and_convert(value):
    return float(value[0])
# Apply the function to the "Actual Value" column
dataset['Actual Value'] = dataset['Actual Value'].apply(remove_brackets_and_convert)
dataset['Predicted Value'] = dataset['Predicted Value'].apply(remove_brackets_and_convert)
#dataset.to_csv('C://Users//CCES-14//Desktop//alm//T//axial//cc//axial_cc_PredVSActual_all.csv')
df1['nn_scf_bs'] = q
df1.to_csv('X_axial_bs_database_all.csv')
## training dataset
import numpy as np
# Load wights file of the best model :
weights_file = 'Weights-819--0.01189-X_axial-bs.hdf5' # choose the best checkpoint 
model.load_weights(weights_file) # load it
# model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
# print("Evaluate model on test data")
# results = model.evaluate(X_test, y_test, batch_size=128)
# print("test loss, test acc:", results)
#computing r2_score for all the actual and predicted output
y_predict_test =  model.predict(X_test)
y_predict_rescale_all = []
min = dfo['scf_bs'].min() #for scf_bs
max = dfo['scf_bs'].max() #for scf_bs
for i in range(0,y_predict_test.size,1):
  y_predict_rescale_all.append((y_predict_test[i]*(max-min))+min)
y_predict_rescale_all=np.array(y_predict_rescale_all)
req = np.array(df_test['scf_bs']) #for scf_bs
req=req[:].reshape(req.size, 1)
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [7.50, 7.50]
plt.rcParams["figure.autolayout"] = True
# p = y[:,0].reshape(y.size, 1)
p = req
q = y_predict_rescale_all[:,0].reshape(y_predict_test.size, 1)
# 100 linearly spaced numbers
x = np.linspace(0,max,100)
# the function, which is y = x^2 here
y = x
plt.title("After rescaling")
plt.scatter(p, q, color="green")
# plot the function
plt.plot(x,y, 'r')
plt.xlabel('y_true')
plt.ylabel('y_predicted')
plt.show()
from sklearn.metrics import r2_score
r2 = r2_score( y_predict_rescale_all[:,0].reshape(y_predict_test.size, 1), req)
print('r2 score for perfect model is', r2)
import pandas as pd
import numpy as np
dataset = pd.DataFrame({'Predicted Value': list(q), 'Actual Value': list(p)}, columns=['Predicted Value', 'Actual Value'])
# Define a function to remove brackets and convert to float
def remove_brackets_and_convert(value):
    return float(value[0])
# Apply the function to the "Actual Value" column
dataset['Actual Value'] = dataset['Actual Value'].apply(remove_brackets_and_convert)
dataset['Predicted Value'] = dataset['Predicted Value'].apply(remove_brackets_and_convert)
#dataset.to_csv('C://Users//CCES-14//Desktop//alm//T//axial//cc//axial_cc_PredVSActual_test.csv')
df_test['nn_scf_bs'] = q
df_test.to_csv('X_axial_bs_database_test.csv')
## training dataset
import numpy as np
# Load wights file of the best model :
weights_file = 'Weights-819--0.01189-X_axial-bs.hdf5' # choose the best checkpoint 
model.load_weights(weights_file) # load it
# model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
# print("Evaluate model on test data")
# results = model.evaluate(X_test, y_test, batch_size=128)
# print("test loss, test acc:", results)
#computing r2_score for all the actual and predicted output
y_predict_val =  model.predict(X_val)
y_predict_rescale_all = []
min = dfo['scf_bs'].min() #for scf_bs
max = dfo['scf_bs'].max() #for scf_bs
for i in range(0,y_predict_val.size,1):
  y_predict_rescale_all.append((y_predict_val[i]*(max-min))+min)
y_predict_rescale_all=np.array(y_predict_rescale_all)
req = np.array(df_val['scf_bs']) #for scf_bs
req=req[:].reshape(req.size, 1)
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [7.50, 7.50]
plt.rcParams["figure.autolayout"] = True
# p = y[:,0].reshape(y.size, 1)
p = req
q = y_predict_rescale_all[:,0].reshape(y_predict_val.size, 1)
# 100 linearly spaced numbers
x = np.linspace(0,max,100)
# the function, which is y = x^2 here
y = x
plt.title("After rescaling")
plt.scatter(p, q, color="green")
# plot the function
plt.plot(x,y, 'r')
plt.xlabel('y_true')
plt.ylabel('y_predicted')
plt.show()
from sklearn.metrics import r2_score
r2 = r2_score( y_predict_rescale_all[:,0].reshape(y_predict_val.size, 1), req)
print('r2 score for perfect model is', r2)
import pandas as pd
import numpy as np
dataset = pd.DataFrame({'Predicted Value': list(q), 'Actual Value': list(p)}, columns=['Predicted Value', 'Actual Value'])
# Define a function to remove brackets and convert to float
def remove_brackets_and_convert(value):
    return float(value[0])
# Apply the function to the "Actual Value" column
dataset['Actual Value'] = dataset['Actual Value'].apply(remove_brackets_and_convert)
dataset['Predicted Value'] = dataset['Predicted Value'].apply(remove_brackets_and_convert)
#dataset.to_csv('C://Users//CCES-14//Desktop//alm//T//axial//cc//axial_cc_PredVSActual_val.csv')
df_val['nn_scf_bs'] = q
df_val.to_csv('X_axial_bs_database_val.csv')
