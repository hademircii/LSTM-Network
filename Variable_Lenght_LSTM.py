
"""
Created on Wed Nov  8 12:19:41 2017

@author: HAD
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import re
import csv
import random
from sklearn.preprocessing import LabelEncoder
from gensim.models import word2vec
from sklearn.utils import shuffle
from nltk import word_tokenize
#%%
#convert label to numeric vector
def label_convert(data):
    enc = LabelEncoder()
    nLabels = len(set(enc.fit(data).classes_))
    data = enc.transform(data)
    return np.eye(nLabels)[data].astype("int32")

#process tweets
def process_line(line):    
    if re.search("http[\S]+", line):
        line = re.sub("http[\S]+","<url>", line)        
    line = re.sub("\Wamp+","and", line)    
    if re.search("[A-Za-z]\W\w+",line):
        line  = re.sub("[^A-Za-z\d\s\'\˘\’\.\,]+"," ", line)
    line = line.lower()   
    cleanLine = word_tokenize(line)
    return cleanLine

#single string for word to vec
def convert_single_string(dataset):
    q = " "
    stringData = []
    for row in dataset:
        line = q.join(row)
        stringData.append(line)
    string = q.join(stringData) 
    return {"single_string" : string, "lines" : stringData }

#%% plotter
def plot(x1, y1, x2, y2, label1 = None, label2 = None, title = None, x_label = None, y_label = None):
    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(x1, y1, color='navy', label = label1)
    ax1.tick_params(bottom='off',top='off',left='off',right='off')
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.set(title= title, ylabel= y_label, xlabel = x_label)
    ax1.plot(x2, y2,color= 'green', label = label2)
    ax1.legend(loc= 'upper center', bbox_to_anchor=(0.5, -0.1),  shadow=True, ncol=2)
    ax1.grid()
    plt.savefig("{}.png".format(title))
    plt.show()
#%%
#read files
f = open('train.csv','r')
trainData = list(csv.reader(f))[1:]
f = open('test.csv','r')
testData = list(csv.reader(f)) [1:]

#shuffle training data 
random.shuffle(trainData)
#%%
#get target and raw input
trainTarget, trainSentences = zip(*trainData) #trainIds
testIds, testTarget, testSentences = zip(*testData) 

noRows = len(trainSentences)
trainTarget = label_convert(trainTarget)
testTarget = label_convert(testTarget)


#%%%
#clean data
cleanTrain = [process_line(row) for row in trainSentences]
cleanTest = [process_line(row) for row in testSentences]

whole = cleanTrain + cleanTest
#%% train word to vec

model = word2vec.Word2Vec(whole, size  = 30, window = 10, min_count=1, iter = 20)

#%%%
maxLength = 35 #max(trainLen)

trainLen = [min(len(each), maxLength) for each in cleanTrain]
testLen = [min(len(each), maxLength) for each in cleanTest]

#%%%
# Train Data
row_count = 0
trainNumeric = np.zeros((len(cleanTrain), maxLength), dtype='int32')

for row in cleanTrain:
    index_count = 0
    for each in row:
        try:
            trainNumeric[row_count][index_count] = model.wv.vocab[each].index
        except:
            trainNumeric[row_count][index_count] = 0
        index_count += 1
        if index_count >= maxLength:
            print("Done with the Row: ", row_count)
            break       
    row_count += 1
    
#Test Data
row_count = 0
testNumeric = np.zeros((len(cleanTest), maxLength), dtype='int32')

for row in cleanTest:
    index_count = 0
    for each in row:
        try:
            testNumeric[row_count][index_count] = model.wv.vocab[each].index
        except:
            testNumeric[row_count][index_count] = 0
        index_count += 1
        if index_count >= maxLength:
            print("Done with the Row: ", row_count)
            break       
    row_count += 1
#%%%%
embeddingMatrix = np.zeros((len(model.wv.vocab), 30), dtype="float32")
for i in range(len(model.wv.vocab)):
    embeddingVector = model.wv[model.wv.index2word[i]]
    if embeddingVector is not None:
        embeddingMatrix[i] = embeddingVector


#%%% Model 1 Parameters
ValidIndexStart = 4199

batchSize = 100
lstmUnits = 35
numClasses = 2

epochs = 15
display_step = 1
#%% define model\
tf.reset_default_graph()

input_data = tf.placeholder(tf.int32, [None, maxLength])

input_length = tf.placeholder(tf.int32, [None])

labels = tf.placeholder(tf.int32, [None, numClasses])

# weight & bias
W =tf.Variable(tf.random_normal([lstmUnits, numClasses]))
b = tf.Variable(tf.random_normal([numClasses]))

data = tf.nn.embedding_lookup(embeddingMatrix,input_data)

lstmCell = tf.contrib.rnn.LSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.MultiRNNCell([lstmCell] * 1)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.5)

value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32, sequence_length=input_length)

word_index = tf.reshape(input_length, [tf.shape(value)[0]])-1

sentence_index = tf.range(0, tf.shape(input_data)[0]) * (tf.shape(input_data)[1])
                      
index = sentence_index + word_index

flat = tf.reshape(value, [-1, tf.shape(value)[2]])
        
last = tf.gather(flat, index)

out_layer = tf.add(tf.matmul(last,W), b)

prediction = tf.nn.softmax(out_layer)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))

accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer,
                                                              labels=labels))

optimizer = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()

#%%%
    
#Run Model
valLoss1 = [] #results for validation loss
trainLoss1 = [] #train loss
valAcc1 = []
trainAcc1 = []

with tf.Session() as sess:
    

    # Run the initializer
    sess.run(init)
    #train model
    ValNumeric, ValLen, ValTarget = trainNumeric[ValidIndexStart+1: ], trainLen[ValidIndexStart+1: ], trainTarget[ValidIndexStart+1: ]  
    trainNumeric_, trainLen_, trainTarget_ = trainNumeric[:ValidIndexStart], trainLen[:ValidIndexStart], trainTarget[:ValidIndexStart]
    
    for epoch in range(epochs):
        nBatch = int( trainNumeric.shape[0] / batchSize )
        trainNumeric_, trainLen_, trainTarget_ = shuffle(trainNumeric_, trainLen_, trainTarget_)
        trainBatchesSentenceSplit = np.array_split(trainNumeric_, nBatch)
        trainBatchesLenSplit = np.array_split(trainLen_, nBatch)
        trainBatchesLabelSplit = np.array_split(trainTarget_, nBatch)
        
        for i in range(nBatch):
            batch = trainBatchesSentenceSplit[i]
            length = trainBatchesLenSplit[i]
            target = trainBatchesLabelSplit[i]
        # Run optimization op (backprop) and cost op (to get loss value)
            _= sess.run([optimizer], feed_dict={input_data: batch,
                                                  input_length: length,
                                                  labels: target
                                                  })
        print("Epoch Done:{}".format(epoch))
        # Display logs per epoch
        if epoch % display_step == 0:
            trainLoss, trainAcc = sess.run([loss, accuracy], feed_dict = {input_data:trainNumeric_,
                                                                          input_length:trainLen_, 
                                                                          labels:trainTarget_         
                                                                           })
            print("-"*60)
            print("Epoch: {}\nTrain Batch:\nLoss:{:0.4f}\n Accuracy: {:0.3f}\n".format(epoch,trainLoss,trainAcc))
            
            # Validate model
            ValLoss, ValAcc = sess.run([loss, accuracy],feed_dict = {input_data: ValNumeric,
                                                                     input_length: ValLen,                               
                                                                     labels: ValTarget 
                                                                    })
            print("Validation Batch:\n Loss:{:0.3f}\n Accuracy:{:0.3f}".format(ValLoss,ValAcc))
            
            trainLoss1.append([epoch,trainLoss])
            trainAcc1.append([epoch,trainAcc])
            valLoss1.append([epoch,ValLoss])
            valAcc1.append([epoch,ValAcc])   
            
   
               
                
    print("Optimization Finished!")
    
    # Test model
    testPrediction, testAcc, testLoss = sess.run([prediction, accuracy, loss], feed_dict = {input_data: testNumeric, input_length: testLen, labels: testTarget} )
    
#%% formatting for plots
x1, y1 = zip(*valLoss1)
x2, y2 = zip(*trainLoss1)
x3, y3 = zip(*valAcc1)
x4, y4 = zip(*trainAcc1)



#%%%
plot(x1, y1, x2, y2, label1 = "Validation Loss", label2 = "Training Loss", title = "Model 1 Loss",y_label = "Loss", x_label = "Epoch")
plot(x3, y3, x4, y4, label1 = "Validation Accuracy", label2 = "Training Loss", title = "Model 1 Accuracy",y_label = "Accuracy", x_label = "Epoch") 
        
#%% get predictions for test data
name = "fp.1"
predictions_test_1 = [row for row in testPrediction]
        
predictions_test_1_id = []        
for i, row in enumerate(predictions_test_1):
    row = row.tolist()
    row.insert(0,i)
    predictions_test_1_id.append(row)
predictions_test_1_id.insert(0, ["id", "realDonaldTrump", "HillaryClinton"])

with open(name + ".csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(predictions_test_1_id)

