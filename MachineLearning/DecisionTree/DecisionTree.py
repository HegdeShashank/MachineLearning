from collections import Counter
from math import log
import pandas as pd
import numpy as np


def Calcuate_Entropy(train_y):
	length=len(train_y)
	counter=Counter(train_y)
	Entropy=0.0
	for key in counter:
		prob=float(counter[key])/length
		Entropy-=prob*log(prob)
	return Entropy

def splitdataset(dataset,axis,value,train_Y):
	retDataSet=[]
	for row_X,row_Y in zip(dataset,train_Y):
		if row_X[axis]==value:

			reducedFeatVec=(row_X[:axis].tolist())
			reducedFeatVec.extend(row_X[axis+1:])
			reducedFeatVec.append(row_Y)
			retDataSet.append(reducedFeatVec)
	return retDataSet

def chooseBestFeature(dataset,train_y):
	numFeaures=len(dataset[0])-1
	bestEntropy=Calcuate_Entropy(train_y)
	bestInfoGain=0.0
	bestFeature=-1

	for i in range(0,numFeaures):
		featList = [example[i] for example in dataset]
		uniqueVals = set(featList)
		newEntropy=0.0
		for value in uniqueVals:
			subset=splitdataset(dataset,i,value,train_y)
			prob=len(subset)/len(dataset)
			labels=[label[-1] for label in subset]
			newEntropy+=prob*(Calcuate_Entropy(labels))
		infoGain=bestEntropy-newEntropy
		if infoGain>bestInfoGain:
			bestInfoGain=infoGain
			bestFeature=i
	return bestFeature

#Function to handle the non-numeric data.
def handle_non_numeric_data(df):
  columns = df.columns.values
  for column in columns:
      text_digit_vals = {}
      def convet_to_int(val):
          return text_digit_vals[val]
      if df[column].dtype != np.int64 and df[column].dtype != np.float64:
          column_content = df[column].values.tolist()
          unique_element = set(column_content)
          x = 0
          for unique in unique_element:
             if unique not in text_digit_vals:
               text_digit_vals[unique] = x
               x += 1
          df[column] = list(map(text_digit_vals.get, df[column]))
  return df


filepath = 'train.csv'
data = pd.read_csv(filepath)
data.convert_objects(convert_numeric=True)
data=handle_non_numeric_data(data)
data.fillna(0, inplace=True)
train_x = data.drop(['Name','PassengerId','Ticket','Survived','Cabin'], 1)
train_Y=data['Survived']
train_x=train_x.values
print(chooseBestFeature(train_x,train_Y))