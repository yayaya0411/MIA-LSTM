import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader import data,wb
import pickle
import warnings
warnings.filterwarnings("ignore")

import configparser
class StockData():
    """
        The process for creating training datasets and evaluation datasets.

        parameter:
            folderPath : The directory path where the stock market stock price csv is stored.
            indexPath : The path of the stock market index main csv file.
            timewindowsize : The amount of time entered into the input of the LSTM. 
                            In the case of 10, the previous 10-day stock price is used to predict the future stock price.
            windowsizeForPCC : The size of the time used to calculate the PCC. In the case of 10, 
                                the previous 10 days stock price is used to find the related stock.
            PositiveStockNumber : Number of positively related stocks..
            NegativeStockNumber : Number of unrelated stocks.
            train_test_rate : Training:Assessment Set Ratio. If it is 0.7, 
                            70% of the generated data set is used for training and 30% is used for evaluation.
            batchSize : The batch size that divides the set.

        process:
            Read the opening and index opening prices of all stocks in the stock market.
            Apply minmax scaler to each stock and index and save.
            PCC calculation, related stock calculation.
            Save dataset

    """
    def __init__(self,folderPath,indexPath,timewindowsize,windowsizeForPCC,PostiveStockNumber,NegativeStockNumber,train_test_rate,batchSize):        
               
        if(train_test_rate<=0 or train_test_rate>=1):
            raise ValueError('train_test_rate should be between 0 and 1')        
        self.P=PostiveStockNumber
        self.N=NegativeStockNumber        
        self.T=timewindowsize
        self.Tr=windowsizeForPCC
        self.folderPath = folderPath      
        self.indexPath=indexPath    
        self.batchSize=batchSize
        # self.date_duration=1700
        self.date_minus=10
        
        self.train_test_rate=train_test_rate
        self.test_rate=0.9
        self.scaler=MinMaxScaler(feature_range=(-1,1))  
        self.indexScaler = MinMaxScaler(feature_range=(-1,1))

        self.indexPrice = self.loadIndex()
        self.stockPrice = self.loadCSV()

        self.trainSet,self.validSet, self.testSet=self.make_dataset()

        self.batchNum={}
    
    def getBatch(self, option):
        """
        Divide the set stored in the class into y,xp,xn,xi,target and create a batch.

        args:
            option='training' or 'evaluation'

        returns:
            batch generator
            batch={'y','xp','xn','xi','target'}
        """
        if(option != 'training' and option != 'valid' and option != 'test'):
            raise ValueError('option should be "training", "valid" or "test".')

        if(option == 'training'):
            # print("Use Training Dataset")
            returnSet = self.trainSet
        if(option == 'valid'):
            # print("Use Valid Dataset")
            returnSet = self.validSet
        if(option == 'test'):
            # print("Use Test Dataset")
            returnSet = self.testSet

        y=[]
        xp=[]
        xn=[]
        xi=[]
        target=[]

        for d in returnSet:
            y.append(d['target_history'])  
            xp.append(d['pos_history'])       
            xn.append(d['neg_history'])       
            xi.append(d['index_history'])       
            target.append(d['target_price'])             
        y=np.reshape(y,(-1,self.T,1))
        xp=np.reshape(xp,(-1,10,self.T,1))
        xn=np.reshape(xn,(-1,10,self.T,1))
        xi=np.reshape(xi,(-1,self.T,1))
        target=np.reshape(target,(-1,1))

        # print(y.shape,xp.shape,xn.shape,xi.shape,target.shape)     

        batchNum=int(len(y)/self.batchSize)
        # self.batchNum[option]=batchNum
        self.batchNum=batchNum

        for i in range(batchNum):
            yield {'y':y[i*self.batchSize:(i+1)*self.batchSize],
                   'xp':xp[i*self.batchSize:(i+1)*self.batchSize],
                   'xn':xn[i*self.batchSize:(i+1)*self.batchSize],
                   'xi':xi[i*self.batchSize:(i+1)*self.batchSize],
                   'target':target[i*self.batchSize:(i+1)*self.batchSize]}

    def loadCSV(self):
        """
        Receives the folder containing the csv file as input and reads the data.
        """
        csvList=os.listdir(self.folderPath)        
        dataframe = pd.DataFrame([])

        for csv in csvList:
            data=pd.read_csv(self.folderPath+'/'+csv,engine='python', encoding= 'unicode_escape')
            #######################
            if(len(data)>self.date_duration):
                data=data[-self.date_duration-1:-1]
                data=data.reset_index()
                data=data['Open']
                dataframe=dataframe.append(data,ignore_index=True)
            
            # data=data[-(self.index_len):-1]
            # data=data.reset_index()
            # data=data['Open']
            # dataframe=dataframe.append(data,ignore_index=True)
            #######################

        dataT=np.array(dataframe).T
        self.scaler.fit(dataT)
        dataT=self.scaler.transform(dataT)
        dataT=dataT.T
        dataframe=pd.DataFrame(dataT)
        dataframe=dataframe.transpose()
        print('StockPrice shape: ',dataframe.shape)
        return dataframe
    
    def loadIndex(self):
        data=pd.read_csv(self.indexPath,engine='python')
        # tune data duration to dataset len
        self.date_duration=data.shape[0]-1
        data=data[-self.date_duration-1:-1]        
        data=data.reset_index()
        data=data.fillna(method='ffill')

        data=np.array(data['Open'])
        data=np.reshape(data,(-1,1))

        data=self.indexScaler.fit_transform(data)

        data=pd.DataFrame(np.squeeze(data))        

        print('IndexPrice shape: ',data.shape)
        return data 


    def make_dataset(self):
        """
        The input and target datasets used in the predictive model.
        The shape of the input data is (target stock + related stock + index, time window size)
        The shape of the target data is (1,1)
        """
        maxday=max([self.T,self.Tr])
        dataset=[]

        # print('maxday:',maxday,'len(self.stockPrice):',len(self.stockPrice) )

        for i in range(maxday,len(self.stockPrice)):
            print('Making dataset progress : {}/{}'.format(i,len(self.stockPrice)),end='\r')
            priceSet=self.stockPrice.loc[i-self.T:i-1]
            targetSet=self.stockPrice.loc[i]
            positiveSet,negativeSet=self.calculate_correlation(self.stockPrice.loc[i-maxday:i-1])
            indexSet = self.indexPrice.loc[i-self.T:i-1]

            for targetNum in priceSet.columns:
                # print('maxday:',maxday,'len(self.stockPrice):',len(self.stockPrice) )
                target_history=np.reshape(np.array(priceSet[targetNum]),(self.T,1))
                pos_history=np.reshape(np.array(positiveSet[targetNum].T),(10,self.T,1))
                neg_history=np.reshape(np.array(negativeSet[targetNum].T),(10,self.T,1))
                index_history=np.reshape(np.array(indexSet),(self.T,1))
                target_price=np.reshape(np.array(targetSet[targetNum]),(1,1))

                dataset.append({'target_history':target_history,
                                'pos_history':pos_history,
                                'neg_history':neg_history,
                                'index_history':index_history,
                                'target_price':target_price,
                                ########
                                'target_num':targetNum,
                            })
        print('Making dataset progress : Finished\t')

        valid_index = int(len(dataset)*self.train_test_rate)
        test_index = int(len(dataset)*self.test_rate)

        trainset = dataset[:valid_index] 
        validset = dataset[valid_index:test_index]
        testset = dataset[test_index:]

        return trainset, validset, testset

    def calculate_correlation(self,priceSet):
        """
        Calculate the Pearson Correlation Coefficient (PCC),
        Creates a set number of positive relational notes in high order and negative relational notes in low order, stores them in the list, and returns.

        The input is the stock price between time windows of all stocks.

        Returns:
        # Relevant stocks of all stocks.
            Positive relationship stock shape = (number of stocks, dataframe(T*P))
            Negative relationship shape = (number of stocks, dataframe(T*N))
        """    
        positive=[]
        negative=[] 
        corr=priceSet[-self.Tr:].corr(method='pearson')

        for i in corr.columns:
            tempCorr=corr[i].sort_values(ascending=False)
            index_P=tempCorr[1:self.P+1].index
            index_N=tempCorr[-self.N:].index
            
            priceSet=priceSet[-self.T:]
            posSet=priceSet[index_P]
            negSet=priceSet[index_N]
            posSet.columns=range(self.P)
            negSet.columns=range(self.N)
            
            positive.append(posSet)
            negative.append(negSet)

        return positive,negative

            
if __name__ == '__main__':
    
    # config = configparser.ConfigParser()
    # config.read('config.ini')
    
    # timesize=config['DATA'].getint('timesize')
    # timesize_for_calc_correlation=config['DATA'].getint('timesize_for_calc_correlation')
    # positive_correlation_stock_num=config['DATA'].getint('positive_correlation_stock_num')
    # negative_correlation_sotck_num=config['DATA'].getint('negative_correlation_sotck_num')
    # train_test_rate=config['DATA'].getfloat('train_test_rate')
    # batch_size=config['DATA'].getint('batch_size')
    
    timesize=20
    timesize_for_calc_correlation=50
    positive_correlation_stock_num=10
    negative_correlation_sotck_num=10
    train_test_rate=0.7
    batch_size=512

    data=StockData(
        'Stock/TW0050','Stock/TWII.csv',
        timesize,
        timesize_for_calc_correlation,
        positive_correlation_stock_num,
        negative_correlation_sotck_num,
        train_test_rate,
        batch_size
        )

    pickle.dump(data, open('data.pk', 'wb'))
    print(f'Dump dataset to ./data.pk')