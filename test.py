from DataSet import *
from Classifer import *
from experiment import *

def test_data(classes,filename,types):
    '''Function to test all DataSet Class and its subsets class methods, if the test for exists case, an addtional input would be 'text' '''
    try:
        classes._datatype == types                                 # test the datatype
        classes._DataSet__load(filename)                            # test the load function
        if filename == "test cases/multiple_choice_responses.csv":  # test the load function for the special case
            classes._data = classes._data.drop(0,axis = 0)          # drop the first row
        classes.clean()                                             # test the clean function
        classes.explore()                                           # test the explore function
        return 1                                                    # return 1 if all tests passed
    except:
        return 0                                                   # return 0 if any test failed


def test_classifer(classes,data,labels,k):
    '''Function to test all Classifier Class and its subsets class methods'''
    try:
        classes.train(data,labels)
        classes.test(data,k)
        print(classes._labels)
        print("In the test of {}, all tests has been passed".format(type(classes).__name__))
    except:
        raise ValueError("the test of {} has failed".format(type(classes).__name__))



if __name__ == "__main__":
    # the test function can be either input by user or use existing test cases
    # if use existing test cases, the input should be "exists"
    '''The test function will be varied based on the version of the project deliverable inputs, for project deliverable please enter 3 for test'''
    version = input("Do project deliverable you want to test (input 2 or 3 or 4 or 5): ")
    if version == "2":
        filename = input("What is the filename, if use existing cases, enter exists: ")
        book = {"quan":QuantDataSet,"qual":QualDataSet,"text":TextDataSet,"time":TimeSeriesDataSet} # book to store the class name
        success = 0                                           # count the number of classes passed all tests
        fail = []                                             # store the classes that are not passed
        if filename == "exists":                              # use existing test cases
            # test cases for data set
            results ={"test cases/yelp.csv":"text","test cases/Sales_Transactions_Dataset_Weekly.csv":"quan","test cases/multiple_choice_responses.csv":"qual","test cases/ptbdb_normal.csv":"time"}
            # loop through the existing test cases
            for key,value in results.items():            #input "text" if asked for input
                try:
                    A = book[value](key)                   # create the class
                    ret = test_data(A,key,value)               # test the class
                    if ret == 1:                          # if all tests passed
                        success += 1                       # add 1 to the success count
                    else:
                        fail.append(book[value])                 # if any test failed, add the file name to the fail list
                except:
                    raise ValueError("the file name didn't match the type")   # if the file name didn't match the type, raise error
        else:
            types = input("What is the type of file (must be one of quan, qual, text, or time ): ")  # ask user to input the type of file if not using existing test cases
            try:
                A = book[types](filename)                   # create the class
                ret = test_data(A,filename,types)                      # test the class
                if ret == 1:                          # if all tests passed
                    success += 1                       # add 1 to the success count
                else:
                    fail.append(book[types])                 # if any test failed, add the file name to the fail list
            except:
                raise ValueError("the file name didn't match the type")  # if the file name didn't match the type, raise error

        print(" There has totally {} classes passed all tests\n The classes that are not passed are {}".format(success,fail)) # print the result

    elif version =="3":
        A = simplekNNClassifier()
        data = [[1,2,3,4],[5,6,7,8],[1,3,6,2],[2,4,6,8],[5,6,7,8],[10,6,3,2]]
        label = [[1],[2],[2],[3],[1],[2]]
        test_classifer(A,data,label,3)
        B = Experiment([[1,2,3,4],[5,6,7,8],[1,3,6,2],[2,4,6,8],[5,6,7,8],[10,6,3,2]],[[1],[2],[2],[3],[1],[2]],[simplekNNClassifier])
        print(B.runCrossVal(3))
        print(B.score())
        print(B._Experiment__confusionMatrix(data,label))
    elif version == "4":
        # test for ROC curve
        np.random.seed(2)
        # Generate random X_train matrix with 5 dimensions and 20 rows
        test_size = 300
        train_size = 800
        X_train = np.random.randint(0,10,size = (train_size,4))
        X_test = np.random.randint(0,10,size = (test_size,4))
        # Generate random labels with equal length
        y_train = [
        [np.random.randint(0, 2)] for i in range(train_size)]
        y_test = [[np.random.randint(0, 2)] for i in range(test_size)]
        A = Experiment(X_train,y_train,[simplekNNClassifier])
        A.ROC(X_test,y_test)


        # test for Decision Tree
        # the test cases are from the example of classification dataframe, the value can be both quantative and qualitive
        data = {'Employed': ["Yes", "No", "No", "No", "Yes", "No", "No", "Yes", "Yes"],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
        'Married': [1, 0, 0, 0, 0, 0, 1, 0, 1]}
        ## Noted: my decision tree will also contrain the column such as "Employed" etc, it only aim to help 
        # understand the tree structure, but it will not be used in the prediction 
        tests = pd.DataFrame({"Employed":["Yes","No","No"],"Gender":["Male","Male","Female"]})
        df = pd.DataFrame(data)
        A = DecisionTreeClassifier()
        A._data = df
        A.train(df,df["Married"])
        print("the Decision tree is",A._tree)
        print("the predicted label of Married for decision tree are",A.test(tests))
        print("the Decision str representation",A)
    elif version == "5":
        data = [[1,2,3,4],[5,6,7,8],[4,3,6,2],[2,4,6,8],[2,8,7,2],[10,6,3,2]]
        label = [[1],[2],[2],[3],[1],[2]]
        test_data = [[1,3,2,4],[6,7,4,1],[1,2,4,1],[2,5,7,2]]
        A = kdTreeClassifier()
        A.train(data,label)
        print("the predicted label for test data is : ",A.test(test_data,2))
    else:
        print("Please input 2 or 3 or 4 or 5")
