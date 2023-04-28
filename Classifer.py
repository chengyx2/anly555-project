
import numpy as np
import pandas as pd
from scipy.stats import mode
from collections import Counter
from math import log2
class ClassifierAlgorithm:

    def __init__(self):
        '''
        initiate the ClassifierAlgorithm

        '''
        self._data = None          # parameter to store training data
        self._labels = None        # parameter to store training labels
        pass


    def train(self):
        '''
        perform training process
        
        '''
        pass # detailed implementation will be implemented in subclass


    def test(self):
        '''
        perform testing process
        
        '''
    
        pass # detailed implementation will be implemented in subclass




class simplekNNClassifier(ClassifierAlgorithm):
    '''Subclass inherited from ClassifierAlgorithm Class, perform simple KNN classification'''
    def __init__(self):
        super().__init__()


    def train(self,data,labels):
        '''
        perform training process
        
        '''
        if data is None or labels is None:
            raise ValueError("data or labels is None")
        
        if isinstance(data,pd.DataFrame):     # make sure data is a pandas dataframe
            self._data = data
            self._labels = np.array(labels)
        else:
            try:                              # try to convert data and labels to pandas dataframe
                self._data = pd.DataFrame(data)
                self._labels = np.array(labels)
            except:
                raise ValueError("data or labels is not a pandas dataframe or can not be converted to pandas dataframe")

    def euclidean(self,point, data):
        '''Euclidean distance between a point  & data'''
        return np.sqrt(np.sum((point - data)**2, axis=1))
   
    def test(self,data,k):
        '''
        perform testing process, which will find the k closest training
        samples and return the mode of the k labels associated from the k closest training
        samples
        
        '''
        data= np.array(data)
        if data.shape[1] != self._data.shape[1]:
            raise ValueError("train and test data dimensions are not matched")
        
        if k > self._data.shape[0]:             # make sure k is not larger than the number of training data
            raise ValueError("k is larger than the number of training data")
        
        train_data = self._data.copy()          # make a copy of training data
        train_label = np.array(self._labels)                  # make a copy of training labels
        neighbors = []
        for x in data:
            distances = self.euclidean(x, train_data)
            dist = np.argsort(distances)[:k]   # find the k closest training samples
            labels = train_label[dist]
            lab = mode(labels) 
            lab = lab.mode[0]
            neighbors.append(lab)
        neighbors = np.array(neighbors)
        self._labels = neighbors
        return neighbors
      

class kdTreeClassifier(ClassifierAlgorithm):
    '''Subclass inherited from ClassifierAlgorithm Class, perform KD Tree classification, the tree will store in an dictionary format, while each level has lable and left and right node information'''
    def __init__(self):
        super().__init__()
        self._tree = {}  # parameter to store the kd tree


    def train(self,data,label):
        ''' Function to build the kd tree, mainly check the input data type and dimension, and call the _buildtree function to build the tree'''
        if not isinstance(data,np.ndarray) or not isinstance(label,np.ndarray):  # check the input data type
            try:
                data = np.array(data)  # try to convert data to numpy array
                label = np.array(label) # try to convert label to numpy array
            except:
                raise ValueError("the input datatype is not valid")  # raise error if can not convert to numpy array
        if len(data) != len(label):             # check the dimension of data and label
            raise ValueError("the data dimension didn't match with label")
        
        results = self._buildtree(data,label,3,depth = 0)       # call _buildtree function to build the tree             
        self._tree = results                                    # store the tree in self._tree parameter
        
    def _buildtree(self,data,label,k,depth):
        '''Function to recursively build the weighted kd tree, the tree will take median of data as node, and split the data into left and right node, and call itself recursively to build the tree, at same time it will also store label information in each level'''
        if len(data) == 1:                                                          # if there is only one data point, return the data point and label
            return {str(data[0])[1:-1]:"None","label":str(label[0,0])}
        elif len(data) == 0:                                                         # if there is no data point, return None
            return "None"
        else:
            axis = depth % k                                                        # calculate the axis to split the data based on method on paper
            depth += 1                                                              # increase the depth by 1                                         
            temp_data = data[:,axis]                                                # get the data on the axis inorder to find the median     
            order = np.argsort(temp_data)
            new_data = data[order]                                                  # sort the data based on the axis                             
            new_label = label[order]                                                # sort the label based on the axis
            medIdx = len(new_data) // 2
            left_data = new_data[:medIdx]                                           # split the data into left and right node                      
            right_data = new_data[medIdx+1:]
            node = new_data[medIdx]
            node_label = new_label[medIdx]                                         #split the the label information in each level
            left_label = new_label[:medIdx]
            right_label = new_label[medIdx+1:]
            left_node = self._buildtree(left_data,left_label,k,depth)               # build the left and right node recursively
            right_node = self._buildtree(right_data,right_label,k,depth)            
            return {str(node)[1:-1]:[left_node,right_node],"label":str(node_label[0])}      # return the node and label information in each level
        

    def test(self,data,k):
        ''''Function to find the predicted label for test data, it will call the _search function to find the path of the test data in the tree, and find the label information in the path, and return the mode of the label information in the path'''
        if not isinstance(data,np.ndarray) or not isinstance(k,int) :  # check the input data type
            try:
                data = np.array(data)
                k = int(k)
            except:
                raise ValueError("the input datatype is not valid")
        
        if k < 1:                                    # check the k value, which can not be smaller than 1
            raise ValueError("k should be larger than 0")
        predictor = []                              # parameter to store the predicted label
        for i in range(len(data)):                  # loop through the test data
            path,path_label = self._search(data[i],self._tree,k,0)      # call _search function to find the path and label_path of the test data in the tree
            nearest = 10                            #set a large number to store the nearest distance
            for j,point in enumerate(path):         # loop through the path to find the nearest distance
                dis = np.sqrt(np.sum((data[i] - point)**2))
                if dis < nearest:
                    nearest = dis
                    temp_label =  path_label[j]
            predictor.append(temp_label)            # append the label information in the path to the predictor parameter
        return(predictor)

    def _search(self,data,tree,k,depth):
        '''Function to create test point path through the weighted tree, it will recursively find the traversal of the test data point, and return the node path and label path information back to test function'''
        if tree == "None" or tree == {}:  # if the tree is empty, return empty string
            return "",""
        else:
            axis = depth % k            # calculate the axis based on method on paper
            path = []                   # parameter to store the node path
            label = []                  # parameter to store the label path
            nodes = list(tree.keys())[0]        # get the node information in each level
            node = np.array(nodes.split(),dtype= int)
            if tree[nodes] == "None":       # if the node is empty, return the node itself and label
                return node,int(tree["label"])
            path.append(node)
            label.append(int(tree["label"]))
            depth += 1
            temp_tree = tree[nodes][0] if data[axis] < node[axis] else tree[nodes][1]       # find the next node based on the test data
            result,ret_label= self._search(data,temp_tree,k,depth)                                    # call itself recursively to find the path
            if isinstance(result,list):                                     # if the result is a list, append the result to the path and label to ensure its one dimensional
                [path.append(i) for i in result if i != ""]
                [label.append(j) for j in ret_label if j != ""]
            else:
                path.append(result) if result != "" else None           # if the result is not a list, append the result to the path and label
                label.append(ret_label) if ret_label != "" else None
            return path,label
        
### Step Count for kdTree Classifier train()
# Let we assume the data is a*b where a is row and b is column(as a numpy matrix) and with total size of N=a*b
# The space complexity for train method is The space complexity is O(N*logN) because the kd tree needs to store all N data points and each level of the tree will split the data into two parts recursively.
# Since there are logN levels, the total number of nodes in the tree is also O(N*logN). Besides, since there has label and node information in each level, so the space complexity is O(2N*logN), which also equal to O(N*logN).
# Since the train() used recursive method, the time complexity is based on the _build_tree() function.
# T(n) = 7(in train function) + 2*T(n/2) + O(n), thus it meet the master theorem case 2, thus the time complexity is also O(N*logN)

### Step Count for kdTree Classifier test()
# Let we assume N is the size of the data and M is the size of the test data
# The space complexity for test method is S(N) = (4 + M + 6) + (2+9+2logN) = O(M) Since the M is the size of the test data which is greater than logN, so the space complexity is O(M)
# Since the test() used recursive method, the time complexity is based on the _search() function.
# T(n) = (5+4m+3m*logN)(test function) + T(n)+O(n)(search function), thus it meet the master theorem case 2, thus the time complexity for search is O(logN)
# However the time complexity for test function is O(M*logN) which is greater than O(logN), so the time complexity for test function is O(M*logN)
        
### Comparison between kdTree Classifier and KNN Classifier
# The Kd tree saved both time complexity compared to KNN classifier. The KNN classifier need to calculate the distance between each test data and all training data, which is O(N*M) time, while KD tree only takes O(M*logN) times, which is less than simple KNN.
# The Kd tree also may have lower space complexity just required O(M) sapce while KNN classifier need to store all training data which is O(N) space, if M< N (in most case) it will save space complexity.



class DecisionTreeClassifier(ClassifierAlgorithm):
    '''Subclass inherited from ClassifierAlgorithm Class, perform Decision Tree classification, this class temprory only support classification decision tree(no regression decision tree)'''
    def __init__(self):
        super().__init__()
        self._tree = {}         # parameter to store the tree
        self._data = None       # parameter to store training data

    def entropy(self,data):
        '''the entropy function for Decision Tree Classifier with input of data'''
        counter = Counter(data)     # count the number of each unique value in data
        ent = 0.0
        for num in counter.values():        # calculate the entropy of data
            p = num / len(data)        # calculate the probability of each unique value
            ent += -p * log2(p)         # calculate the entropy of data
        return ent

    def infogain(self,data,labels):
        '''the information gain function for Decision Tree Classifier with input of label'''
        entroS = self.entropy(labels)           # calculate the entropy of labels
        unq = labels.unique()               # find the unique values in labels
        ens = {}                            # create a dictionary to store the entropy of each feature
        for i in unq:                       # calculate the entropy of each feature
            subset = data[labels == i]      # subset the data based on the unique value in labels
            subset = subset.drop(labels.name, axis=1)       # drop the labels column
            ent = []                            # create a list to store the entropy of each feature
            for j in subset.columns:
                ent.append(self.entropy(subset[j])*(subset.shape[0]/data.shape[0]))     # calculate the entropy of each feature
            ens[i] = ent
        ens = pd.DataFrame(ens, index=subset.columns).transpose()       # convert the dictionary to a dataframe
        Sf = {}
        for names in ens.columns:
            Sf[names] = entroS - ens[names].sum()
            
        return Sf
    
    def _predict(self,data,labels):
        '''the tree building function for Decision Tree Classifier with input of data and labels'''
        if data is None or labels is None:          # make sure data and labels are not None
            raise ValueError("data or labels is None")      
        IGGs = self.infogain(data,labels)           # calculate the information gain of each feature
        if len(IGGs) <= 1 or len(labels.unique()) == 1:     # if there is only one feature or all the labels are the same, return the mode of labels
            temp = {}                                # create a dictionary to store the mode of labels
            split = list(IGGs.keys())[0]            # find the feature with the highest information gain
            for i in data[split].unique():              # find the mode of labels for each unique value in the feature
                subset = data[data[split] == i]
                temp_label = labels[data[split] == i]       # subset the labels based on the unique value in the feature
                temp[i] = temp_label.mode()[0]          # find the mode of labels for each unique value in the feature
        else:
            split = max(IGGs, key=IGGs.get)             # else if the S is not empty find the feature with the highest information gain
            temp = {}       
            for i in data[split].unique():
                subset = data[data[split] == i]         # subset the data based on the unique value in the feature
                subset = subset.drop(split, axis=1)     # drop the feature with the highest information gain
                temp_label = labels[data[split] == i]       # subset the labels based on the unique value in the feature
                temp[i] = self._predict(subset,temp_label)     # recursively call the train function to build the tree
            
        
        return {split:temp}

    def train(self,data,labels):
        ''' the train function for Decision Tree Classifier with input of data and labels'''
        result = self._predict(data,labels)         # call the _predict function to build the tree

        self._tree = result                          # store the tree in the parameter
        return result                               # return the tree

    def test(self,data):
        labels = []                                 # create a list to store the predicted labels
        for i in range(data.shape[0]):              # loop through the test data to predict the labels
            labels.append(self._test_predict(data.loc[i].to_dict(),self._tree))     # call the _test_predict function to predict the labels
        
        return labels

    def _test_predict(self,data,tree):
        ''' the test recursive function for Decision Tree Classifier with input of data and tree'''
        value = list(data.values())                             # get the values of the test data
        for node in tree.keys():                            # loop through the tree to find the predicted label
            if not isinstance(tree[node],dict):             # if the node is not a dictionary, return the label
                ret = list(tree.values())[0]                # return the label
            else:
                if node in data.keys():                       # if the node is in the test data, recursively call the _test_predict function
                    ret = self._test_predict(data,tree[node])       # recursively call the _test_predict function
                if node in value:                        # if the node is in the test data, recursively call the _test_predict function
                    name = {i for i in data if data[i]==node}                   # find the name of the node
                    temp_data = {i:data[i] for i in data if i != name}      # create a new dictionary to store the test data without the node
                    ret = self._test_predict(temp_data,tree[node])          # recursively call the _test_predict function
        return ret

    def _tostring(self,tree):
        ''' the recursive helper function to convert the tree to a string'''
        total = ""                  # create a string to store the tree             
        for key in tree:            # loop through the tree to convert the tree to a string
            strs = "["+ str(key)            # convert the tree to a string
            if isinstance(tree[key],dict):    # if the node is a dictionary, recursively call the _tostring function
                strs += self._tostring(tree[key])       # recursively call the _tostring function
            else:
                strs += "["+str(tree[key])+"]"
            strs += "]"
            total += strs
        return total
    
    def __str__(self):
        ''' the function to convert the tree to a string'''
        return (str(self._tostring(self._tree)))       # call the _tostring function to convert the tree to a string
                

### Step Count for simple kNN Classifier
# Let we set the input data for space a*b (as a dataframe), and k for space 1. Thus, since we need to compute
# the space count of S(n), we assume n = a*b where a is row and b is column
# S(n) = 4+ n(the copy of data) + n/a(the labels) + 6 + n(new lists created) +2 =  2n + 2n/a + 12  which is O(n^2) since the biggest term is n, which is the size of data
# T(n) = 7 + 7n(seven operation in the for loops) + 3 = 7n + 10 +m which is O(m*n) since the biggest term is n, which is the size of data

### Step Count for Decision Tree Classifier
# Let we set the input data for space a*b where a is row and b is column (as a dataframe).
# The step count for the train and test function are little complex, it need also consider the count in the helper functions, which requires extra space.
# The space count of S(n) = 5 (space for entropy)+ (a*b + 7+ 2*a)(space for infog)+ b*(8+2*a)(space for _predict) + 4+a(space for train and test) + a*(8+2*b)(space for _test_predict) 
# S(n) = 5+a*b+7+2a+8b+2a*b+4+9a + 8a+ 2a*b = 5a*b + 17a + 8b+ 16, which is O(a*b) since the biggest term is a*b, which a*b is the size of data.
# T(n) = 5 +2 + 4*log(a*b) (for the recursion) + 3 + 7*log(a*b) + 3 + 1 + 2*a + 1 + 6*log(a*b)+1, which is O(log(a*b)) since the biggest term is log(a*b), which a*b is is the size of data.
# noted: since the input size n = a*b, thus, the space complexity is O(n) and time complexity is O(log(n))

 

if __name__ == "__main__":
    data = [[1,2,3,4],[5,6,7,8],[4,3,6,2],[2,4,6,8],[2,8,7,2],[10,6,3,2]]
    label = [[1],[2],[2],[3],[1],[2]]
    test_data = [[1,3,2,4],[6,7,4,1],[1,2,4,1]]
    A = kdTreeClassifier()
    A.train(data,label)
    print(A.test(test_data,2))