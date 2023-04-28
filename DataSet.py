
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import wordcloud
import random


class DataSet:

    def __init__(self,filename):
        '''Initiate the dataset class with the name of a csv file'''
        # add code in future deliverable
        self._data = None  # create variable to store the data, which should be a dataframe
        self._filename = filename
        self._datatype =None  #determine the input file type

    def __readFromCSV(self,filename):
        '''
        read csv or txt file with filename
        '''
        try:
            if self._datatype == "csv":           
                data = pd.read_csv(filename)             #read data with pandas csv
            elif self._datatype == "txt":
                data = pd.read_csv(filename, sep=" ",header= None)  #read data with pandas csv but remove header and add sep
            
            return data
        except:
            raise Exception('Please enter the filename end with .csv or .txt')

    def __load(self,filename):
        '''
        load the file with filename and determine its type whether be CSV or TXT
        '''
        if '.csv' in filename:                            #assign the datatype as csv if its a csv file 
            self._datatype = "csv"
            self._data = self.__readFromCSV(filename)
        elif '.txt' in self._filename:                     #assign the datatype as csv if its a txt file 
            self._datatype = "txt"
            self._data  = self.__readFromCSV(filename)
           
        else:                                              #other wise its not a supported datatype, raise error
            raise Exception('Please enter the filename end with .csv or .txt')

    def clean(self):
        '''
        perform data cleaning to loaded dataset
        '''
        pass

    def explore(self):
        '''
        perform eda to the cleaned dataset
        '''
        pass






#From here are the subclasses for Dataset Class, The Dataset will be the parent class
# for all the following classes

class TimeSeriesDataSet(DataSet):
    '''Subclass inherited from DataSet Class, Defined TimeSeries format Dataset'''
    def __init__(self,filename):
        super().__init__(filename)
        self._type = "time"                        #set the data type as time
        self.window_size = 6                       #set the window size 

    def median_filter(self,temp):
        """Function to filter the data with median filter."""
        data = np.zeros(len(temp))                  #create a numpy list with all zeros
        for i in range(len(temp)):                  #apply the windowsize median to all value in the data
            window_start = max(0, i - self.window_size//2)
            window_end = min(len(temp), i + self.window_size//2 + 1)
            data[i] = np.median(temp[window_start:window_end])       
        return data
    
    def clean(self):
        """ run a median filter that removes the outliers of data"""
        try:
            ret = self._data.to_numpy()          # inorder to have a faster speed, convert pandas to numpy since numpy is really fast
            for i in range(len(ret)):            # loop every inner array in the nested numpy array for median filter operation
                ret[i] = self.median_filter(ret[i])

            self._data = pd.DataFrame(ret)       # convert back to dataframe to unity the type of self._data
        except:
            raise ValueError("invalid data type, please check the data type of the dataset")
        
    def explore(self):
        """plot the data with matplotlib with """
        try:
            x = random.randint(0,len(self._data.columns))   #randomly select a column to plot
            y = random.randint(0,len(self._data.columns))   #randomly select a column to plot
            ret = self._data.copy()              #copy the data to a new variable
            fig,ax = plt.subplots(figsize=(10,8))   #create a figure and axis
            ax.hist(ret.iloc[:,x],bins =20, color = "#12664F", edgecolor = "black")
            ax.set_title("Histogram of column "+str(x))
            plt.show()                            #show the plot

            fig,ax = plt.subplots(figsize=(10,8))   #create a figure and axis
            ax.scatter(ret.iloc[:,y],ret.iloc[:,x],color = "#2DC2BD", edgecolor = "black")
            ax.set_title("Scatter plot of column "+str(x)+" and column "+str(y))
            plt.show()                            #show the plot
        except:
            raise ValueError("invalid data type, please check the data type of the dataset")

class TextDataSet(DataSet):
    '''Subclass inherited from DataSet Class, Defined Text format Dataset'''
    def __init__(self,filename):
        super().__init__(filename)
        self._type = "text"                        #set the data type as text
        self._text = input("please enter the column you want to clean and explore") #ask user to enter the column name


    def clean(self):  
        """clean the text data by removing stopwords, punctuation, and lemmatize the data"""
        ret = self._data.copy()
        stopwords=nltk.corpus.stopwords.words('english')
        wnl = nltk.WordNetLemmatizer()
        tmp = nltk.PorterStemmer()
        select = self._text
        try: 
            # remove stopwords in the data 
            ret[select] = ret[select].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]) )
            # remove punctuation in the data
            ret[select]=ret[select].str.replace('[^\w\s]','')
            # tokenize the data into lists
            ret[select]=ret[select].apply(lambda x: re.split('\W+', x.lower()))
            # stem the tokenized data
            ret[select] = ret[select].apply(lambda x: [tmp.stem(word) for word in x])
            # lemmatize the tokenized data
            ret[select] = ret[select].apply(lambda x: [wnl.lemmatize(word) for word in x])
            # join the tokenized data back to string
            ret[select] = ret[select].apply(lambda x: " ".join(x))
        except:
            # if the column name is not valid, raise error
            raise ValueError("invalid column to clean or column name not exists")
            
        # assign the cleaned data to self._data
        self._data = ret
        
    def explore(self):
        """plot the data with matplotlib with pie chart (top 30 words) and wordcloud """
        try:
            ret = self._data.copy()
            text =  pd.Series(' '.join(ret[self._text]).lower().split()).value_counts()[:30]
            text2 = " ".join(i for i in ret[self._text])
            # First graph is wordcloud
            wordcloud_gen=wordcloud.WordCloud(background_color='white',colormap='Pastel1').generate(text2)
            plt.figure( figsize=(15,10))
            plt.imshow(wordcloud_gen)
            plt.show()
            # Second graph is a bar plot which generate the highest 30 frequent words
            fig,ax=plt.subplots(figsize= (10,8))
            ax.barh(text.keys(),text.values)
            ax.set_title('Top 10 words')
            plt.show()
        
        except:
            raise ValueError("invalid column to clean or column name not exists")
        


class QuantDataSet(DataSet):
    '''Subclass inherited from DataSet Class, Defined Quantity format Dataset'''
    def __init__(self, filename):
        super().__init__(filename)
        self._type = "quan"                        #set the data type as quantity

    def clean(self):
        """clean the data by filling the missing value with mean of the column"""
        results = self._data.copy()                #copy the data to a new variable
        
        for i in results.columns:                  #loop every column in the data
            try:
                results[i] = results[i].astype(float) #try to convert the column to float
            except:
                results = results.drop([i],axis=1)   #if the column cannot be converted to float, drop the column

        if results.shape[1] == 0:                    #if the data is empty, raise error
            raise ValueError("invalid data set for clean")
        means = results.mean()                     #calculate the mean of the data
        results = results.fillna(means)            #fill the missing value with mean of the column
        self._data = results                       #assign the cleaned data to self._data
        
        

    def explore(self):
        """plot the data with matplotlib with scatter plot and histogram"""
        try:
            xs = random.choice(self._data.columns)     #randomly select a column to plot
            ys = random.choice(self._data.columns)     #randomly select a column to plot
        
            fig,ax=plt.subplots()                             #create a figure and axis
            ax.scatter(self._data[xs],self._data[ys],color = "#963484")         #plot the scatter plot
            fig.set_size_inches(10,6)
            ax.set_xlabel(xs,fontsize=15)
            ax.set_ylabel(xs,fontsize=15)
            ax.set_title("{} vs {}".format(xs,ys),fontsize=15)
            plt.show()

            fig,ax=plt.subplots()
            ax.hist(x=self._data[xs],bins = 20,color="#C5DCA0")            #plot the histogram
            fig.set_size_inches(10,6)
            ax.set_xlabel(xs,fontsize=15)
            ax.set_title("The distribution of {}".format(xs),fontsize=15)
            plt.show()
        
        except:
            raise ValueError("invalid data type, please check the data type of the column")
        



class QualDataSet(DataSet):
    '''Subclass inherited from DataSet Class, Defined Quality format Dataset'''

    def __init__(self, filename):
        super().__init__(filename)
        self._type = "qual"                        #set the data type as quality

    def clean(self):
        '''fill the missing values with the median or mode, if the columns value can convert to numeric then use mean otherwise use mode'''
        ret = self._data.copy()           #get the copy of dataframe
        nums = []                         #create list that hold column name for all value able to convert to int
        modes = []                        #create list that hold column name for columns can't convert to int
        for i in ret.columns:
            try:
                ret[i].astype(float)      #append columns names that can convert to flaot
                nums.append(i)
            except:
                modes.append(i)           #append column names that cant convert to float

        for j in nums:                    #fill missing value with column mean
            ret[j] = ret[j].astype(float)  
            means = ret[j].mean()          #calculate mean
            ret[j] = ret[j].fillna(means)
        
        for n in modes:                    #fill missing value with column modes   
            one = ret[n].mode(dropna = True)[0]           #get the mode
            ret[n] = ret[n].fillna(one)   #fill the missing value with mode
        
        self._data = ret                    #assign final dataframe to self._data

       
    def explore(self):

        x =random.choice(self._data.columns)   #randomly select a column to plot
        if x not in self._data.columns:              # if column not exist raise error
            raise ValueError("invalide column name")

        try:
            fig, ax = plt.subplots(figsize=(15, 9))     #create a hist plot 
            ax.hist(x=self._data[x],bins = 20,color="#A7A7A9") #plot the hist
            ax.set_xlabel(x,fontsize=15)
            ax.set_title('Qualitative histogram for {}'.format(x),fontsize=15)
            plt.show()
        
        except:
            raise ValueError("invalid data type for histogram, please check the data type of the column")

        try:
            fig, ax = plt.subplots(figsize=(15, 9))     #create a pie plot
            self._data.groupby(x).size().plot(kind='pie', textprops={'fontsize': 20}, 
                                  colors=['tomato', 'gold', 'skyblue'],ax = ax)  #plot the pie
            ax.set_title('Qualitative pie plot for {}'.format(x),fontsize=15)
            plt.show()
        except:
            raise ValueError("invalid data type for pie plot, please check the data type of the column")
        

        


 
    