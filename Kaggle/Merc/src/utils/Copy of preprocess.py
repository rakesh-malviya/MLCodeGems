import pandas as pd

"""Categorical Data Handling"""
def dfLabelEncoding(df_train,df_test):
  from sklearn.preprocessing import LabelEncoder
  le=LabelEncoder()
  for col in df_test.columns.values:
    if df_test[col].dtypes=='object':
      data = df_train[col].append(df_test[col])
#       print(data)
      le.fit(data.values)
      df_train[col]=le.transform(df_train[col])
      df_test[col]=le.transform(df_test[col])
      
  return df_train,df_test
      

def dfOneHotEncoding(df_train,df_test,columnList):
  from sklearn.preprocessing import OneHotEncoder
  enc=OneHotEncoder(sparse=False)
  for col in columnList:
       # creating an exhaustive list of all possible categorical values
#        print("col:",col)       
       data=df_train[[col]].append(df_test[[col]])
       enc.fit(data)
       # Fitting One Hot Encoding on train data
       temp = enc.transform(df_train[[col]])
       # Changing the encoded features into a data frame with new column names
       temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]
            .value_counts().index])
       # In side by side concatenation index values should be same
       # Setting the index values similar to the df_train data frame
       temp=temp.set_index(df_train.index.values)
       # adding the new One Hot Encoded varibales to the train data frame
       df_train=pd.concat([df_train,temp],axis=1)
       # fitting One Hot Encoding on test data
       temp = enc.transform(df_test[[col]])
       # changing it into data frame and adding column names
       temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]
            .value_counts().index])
       # Setting the index for proper concatenation
       temp=temp.set_index(df_test.index.values)
       # adding the new One Hot Encoded varibales to test data frame
       df_test=pd.concat([df_test,temp],axis=1)  
  
#   print(len(df_test.columns))  
  df_train.drop(columnList, axis=1, inplace=True)
  df_test.drop(columnList, axis=1, inplace=True)  
#   print(len(df_test.columns))
  
  return df_train,df_test
  
  
                