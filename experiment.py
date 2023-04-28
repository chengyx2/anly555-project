import pandas as pd
import numpy as np
from Classifer import *
import matplotlib.pyplot as plt
import math  

class Experiment:
    def __init__(self, data, labels, classifier):
        '''initiate the Experiment class with input of data, labels and classifier'''
        if isinstance(data,pd.DataFrame):     # make sure data is a pandas dataframe
            self._data = data
            self._labels = np.array(labels)
        else:
            try:                              # try to convert data and labels to pandas dataframe
                self._data = pd.DataFrame(data)
                self._labels = np.array(labels)
            except:
                raise ValueError("data or labels is not a pandas dataframe or can not be converted to pandas dataframe")

        self._classifier = classifier if isinstance(classifier,list) else list(classifier)
        for c in self._classifier:                  # check if all classifiers in the list can fit the dataset and lables, if not, raise error
            try:
                ret = c()
                ret.train(self._data,self._labels)
            except:
                raise ValueError("{} is not a valid classifier".format(c))
      
    def runCrossVal(self, k):
        '''perform the cross validation for Experiemtn Class'''
        
        if not isinstance(k,int) or k >= 20:  # set the limit for k to be less than 20 since if k too large, the computation time will eslcated geometrically
            raise ValueError("k must a integer and less than 20")
        
        results = [] # create a matrix to store the results
        ret = self._data.copy()
        index = [int((len(ret)/k)*n) for n in range(k)]   # create a list of index to split the data
        index.append(len(ret)) 
        for n in range(len(index)-1):                       # loop through the index list to split the data
            temp = ret.copy()                               # create a copy of the data to avoid changing the original data
            temp_labels = self._labels.copy()               # create a copy of the labels to avoid changing the original labels
            test_data = temp.iloc[index[n]:index[n+1],:]    # split the data into test data and train data
            train_data = temp.drop(temp.index[index[n]:index[n+1]])                 #get the train data
            temp_labels = np.delete(temp_labels,range(index[n],index[n+1]))         # get the train labels
            temp_labels = [[x] for x in temp_labels]                             # convert the labels to a list of list
            for model in self._classifier:                    # loop through the classifier list to get the results
                c = model()
                c.train(train_data,temp_labels)
                c.test(test_data,k)
                results.append(c._labels)
        return np.array(results)


    def score(self):
        ''' get the score for the experiement class '''
        # add code in future deliverable
        results ={}                         # create a dictionary to store the results                                      
        for model in self._classifier:      # loop through the classifier list to get the results
            c = model()                     # initiate the classifier
            c.train(self._data,self._labels)
            if len(self._data) < 5:         # if the data is less than 5, use all the data to test the model
                c.test(self._data, k = len(self._data))
            else:
                c.test(self._data, k = 5)      # use 1/3 K neighbors to test the model
            success = sum([1 if i == j[0] else 0 for i,j in zip(c._labels,self._labels)])         # loop through the predicted labels and actual labels to get the score
            
            results[model] = success/len(self._labels)      # store the results in the dictionary


        return pd.DataFrame([results])


    def __confusionMatrix(self,data,labels):
        ''' generate the confusion matrix for the experiement class '''
        # add code in future deliverable
        for model in self._classifier:                  # loop through the classifier list to get the results
            nums = np.unique(labels)          # get the unique labels
            matrix = np.zeros((len(nums),len(nums)))        # create a matrix to store the results
            c = model()                                     # initiate the classifier
            c.train(data,labels)            # train the model   
            c.test(data, 5)      # use 1/3 K neighbors to test the model
            c._labels = [x[0] for x in c._labels]           # convert the predicted labels to a list\
            df = pd.DataFrame([c._labels,labels]).transpose()    # convert the predicted labels and actual labels to a dataframe
            df.columns = ['Predicted','Actual']             # rename the columns
            for i in range(len(nums)):                      # loop through the unique labels to get the results
                for j in range(len(nums)):
                    temp = df[(df['Predicted'] == nums[i])]
                    matrix[i,j] = temp[temp["Actual"] == nums[j]].shape[0]      # store the results in the matrix
        return matrix


    def predict_proba(self, test_data,k):
        ''' generate the predicted probability for the experiement class '''
        dicts = []
        for i in test_data:
            distance = np.sqrt(np.sum((i - self._data)**2, axis=1))
            dist = np.argsort(distance)[:k]   # find the k closest training samples
            labels = self._labels[dist]       # get the labels of the k closest training samples
            ret = []                         # initiate the probability
            for name in np.unique(self._labels):  # loop through the unique labels to get the probability
                prob = np.sum(labels == name)/len(dist)  # calculate the probability
                ret.append(prob)
            dicts.append(np.array(ret))   # store the results in the dictionary
        return np.array(dicts)
    
    
    def ROC(self,test_data,test_labels):
        ''' generate the ROC curve for the different classifier, the ROC curve will be assume this is a binary classifer label, which assume second label is the postivie, the ROC curve looks slightly better than random classifer may becasue of the dataset: no relationship between label and vectors.'''
        for model in self._classifier:                  # loop through the classifier list to get the results
            c = model()                                 # initiate the classifier
            c.train(self._data,self._labels)            # train the model
            c.test(test_data, 7)      # use 1/3 K neighbors to test the model
            matx = np.sum(self.__confusionMatrix(test_data,test_labels),axis=0)
            pos = matx[0]
            neg = matx[1]
            probs = self.predict_proba(test_data, 7)
            roc = np.array([])
            probs = probs[:,0] # get the probability of the second positive label    
            values = {i:j for i,j in enumerate(probs)} # create a dictionary with the index and probability
            labs = {i:j[0] for i,j in enumerate(test_labels)} # create a dictionary with the index and label
            sorted_values = dict(sorted(values.items(), key=lambda item: item[1]))
            sorted_keys  = list(sorted_values.keys()) # sort the dictionary by the probability
            i,fp,tp = 0,0,0  # initiate the false positive, true positive and index
            fprev = 1     # initiate the previous probability
            while i < len(sorted_values): # loop through the sorted probability
                if sorted_values[i] != fprev:       # if the probability is different from the previous one, then we need to calculate the ROC
                    roc = np.append(roc, [fp/neg, tp/pos])  # append the ROC to the array
                    fprev = sorted_values[i]        # update the previous probability
                idx = sorted_keys[i]        # get the index of the probability
                if labs[idx] == 0:      # if the label is 0, then it is a true positive
                    tp += 1
                else:
                    fp += 1
                i += 1
            roc = np.append(roc, [fp/neg, tp/pos])      # append the last ROC (1,1) to the array
            roc = roc.reshape(-1, 2)            # reshape the array to 2 columns
            plt.plot(roc[:,0],roc[:,1],linewidth=2,label = model.__name__) # plot the ROC curve
        plt.title('ROC Curve',fontsize=20)                # plot the ROC curve
        plt.xlabel('False Positive Rate',fontsize=16)    # plot the ROC curve
        plt.legend()
        plt.ylabel('True Positive Rate',fontsize=16)
        plt.show()
### Step Count for Score()
# since there has no input data for score(), thus we set n as the size of self._labels
# S(n) = 1+ 5 + n(new dictionary created with length of n) + 2=  n + 8 which is O(n) since the biggest term is n, which is the size of data
# For time complexity, the n equal to the size of classifier list, which is 1 in this case
# T(n) = 5n(5 operation in the for loop)+2  which is O(n) since the biggest term is n, which is the size of data


### Step Count for confusion matrix()
# since there has no input data for score(), thus we set n as the size of the unique labels
# S(n) = 1+ n(new lists with size of n) + n*n(new matrix created with length of n) + 3 + 2n(dataframe size) + 6(the all following operations)=  n*n + 3n + 10 which is O(n*n) since the biggest term is n*n, which is the size of unique labels
# For time complexity, the n also equal to the size of the unique labels
# T(n) = 8n(8 operation in the for loop)+n*n + 4n^3 = 4n^3 + n^2 + 8n,  which is O(n^3) since the biggest term is n^3, which is the size of data
# noted: Although the time and space complexity seems to be large for this method, considering the size of 
# the unique labels is usually small compare to size of the data, thus the time and space complexity are not that bad.


### Step Count for ROC()
#the ROC has the input of test_data and test_labels, thus we set n as the size of test_data, the test_data usually be a matrix with size a*b where a is rows and b is column
# and the size of test_labels is a since the test_label is one dimensional.
#S(n) = O(n= a+b)(the sapce required for the simpleKNNClassifier) +24 + 2a (length of sorted list) +10+a(space for store the probability) 
# S(n) = O(n) since the biggest term is n = a*b used in classifer, which is the size of data a*b
# For time complexity, the input n is also a*b T(n) = 2+8*a + 2*a^(a/2) (depend on unique label size)+ O(n)(time complexity to train and test model) + 8+ 10 + 9*a 
# Thus, T(n) = O(a^2) for average cases. Since the biggest term is a^2, the O(a^2) may greater or smaller than O(n) depends on the size of b.

    
    
