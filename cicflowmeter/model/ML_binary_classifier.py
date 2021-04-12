# %%
#import statements
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import sys
from pickle import dump

task = sys.argv[1]
if task == 'train':
    train_flag = True
elif task == 'test':
    train_flag = False
    model_path = sys.argv[4]
else:
    sys.exit('Invalid Operation')

features_file = sys.argv[2]
run_no = sys.argv[3]
print('running script run no: ' + sys.argv[3])

max_val = 99999

with open(features_file) as f:
    features = [feature.strip() for feature in f]

if train_flag:
    with open('train_files.txt') as f:
        train_files = [filename.strip() for filename in f]

with open('test_files.txt') as f:
    test_files = [filename.strip() for filename in f]

if train_flag:
    train_dir = 'input/CSV-01-12/01-12/'

test_dir = 'input/CSV-03-11/03-11/'
output_dir ='./output/'
# @TODO create preprocessing transforming model
scaler = QuantileTransformer(n_quantiles=1000, random_state=42)
# %%
#load dataset into dataframe
def read_file(filename, y_out):
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    df = df[features]
    NewLabel = []
    for i in df["Label"]:
        if i =="BENIGN":
            NewLabel.append(0)
        else:
            NewLabel.append(1)
    df["Label"]=NewLabel
    y = df['Label'].values
    y_out = y_out.extend(y)
    del df['Label']
    df = df.replace('Infinity', max_val)
    x = df.values
    # scaled_df = scaler.fit_transform(x)
    # x = pd.DataFrame(scaled_df)
    return x
    
nClasses = 2
#### LOAD TRAIN DATA ######
if train_flag:

    temp_y = []

    original_x = None
    print(type(original_x))
    for f in test_files:
        print('Processing file ' + f + '\n')
        if original_x is None:
            original_x = read_file(test_dir + f, temp_y)
        else:
            original_x = np.concatenate((original_x, read_file(test_dir + f, temp_y)))
        print('Processed file ' + f + ' , total samples is ' + str(len(temp_y)) + '\n')
    # transform
    transformed_x = scaler.fit_transform(original_x)
    new_x = pd.DataFrame(transformed_x);
    #new_y = np.asarray(temp_y)
    new_y = to_categorical(temp_y, num_classes=nClasses)

    xTrain, xVal, yTrain, yVal = train_test_split(new_x, new_y, test_size = 0.2, random_state = 42)
    #np.savetxt('train_idx', idx1)
    #np.savetxt('test_idx', idx2)

    print('train size: ', xTrain.shape)
    print('train labels: ', yTrain.shape)
    print('Valid size: ',  xVal.shape)

#### LOAD TEST DATA #######
new_x = pd.DataFrame()
temp_y = []
original_x = None
print(type(original_x))
for f in test_files:
    print('Processing file ' + f + '\n')
    if original_x is None:
        original_x = read_file(test_dir + f, temp_y)
    else:
        original_x = np.concatenate((original_x, read_file(test_dir + f, temp_y)))
    print('Processed file ' + f + ' , total samples is ' + str(len(temp_y)) + '\n')
# @TODO tranform or fit_trainform
transformed_x = scaler.fit_transform(original_x)
new_x = pd.DataFrame(transformed_x)
print(original_x.shape)
print(len(temp_y))
# @TODO save scaler
dump(scaler, open('output/scaler.pkl', 'wb'))

xTest = np.asarray(new_x)
#yTest = np.asarray(temp_y)
yTest = to_categorical(temp_y, num_classes=nClasses)

#xTest, xTemp, yTest, yTemp = train_test_split(new_x, new_y, test_size = 0.01, random_state = 42)
#np.savetxt('train_idx', idx1)
#np.savetxt('test_idx', idx2)

print('test size: ',  xTest.shape)



model = Sequential()
model.add(Dense(64, input_dim=len(features)-1, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(nClasses, activation='softmax'))
print(model.summary(90))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

num_batch = 1000
num_epochs = 10
es = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=num_epochs, mode='auto', baseline=None, restore_best_weights=True, verbose=1)
if train_flag:
    model.fit(xTrain, yTrain, batch_size=num_batch, validation_data=[xVal, yVal], epochs = num_epochs, verbose=1)
    #, callbacks=[es])
    model.save(output_dir + run_no + '/model_weights')
else:
    model = load_model(model_path)
print(model.evaluate(xTest,yTest))

prediction = np.argmax(model.predict(xTest), axis=1)
y_test = np.argmax(yTest, axis=1)
print(metrics.classification_report(y_test, prediction))
np.savetxt(output_dir + run_no +'/predictions.txt', prediction)
np.savetxt(output_dir + run_no +'/ground_truth.txt', y_test)








