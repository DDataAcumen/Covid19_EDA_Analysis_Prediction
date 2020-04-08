# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 13:18:02 2020

@author: DemiiGod
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

dataset = pd.read_csv('train.csv')
dataset = pd.DataFrame(dataset)
#dataset.columns = 'Id','Province_State','Country_Region', 'Date', 'ConfirmedCases', 'Fatalities'
from pandas.plotting import scatter_matrix
scatter_matrix(dataset, diagonal = 'kde')
dataset.dtypes

dataset.loc[dataset['Province_State'].isnull(),'Province_State'] = dataset['Country_Region']

#Changing Datatypes as per the specification
dataset[['Province_State', 
         'Country_Region']] = dataset[['Province_State', 
         'Country_Region']].astype('category')
dataset[['ConfirmedCases', 'Fatalities']] = dataset[['ConfirmedCases', 'Fatalities']].astype(int)                                       
dataset['Date'] = pd.to_datetime(dataset.Date)
#dataset['Date'] = dataset.Date.strip(when, '%Y-%m-%d').date()
#dataset['Date'] = dataset['Date'].strftime("%x")
dataset.dtypes

# Replacing nan Values in States
# dataset.Province_State = dataset.Province_State ['Province_State']isnull == True].mask(dataset['Country_Region'])  
# State= dataset.Province_State.fillna(dataset.Country_Region, inplace=True)
#  def f(x):
#     if np.isnan(x['Province_State']):
#         return x['Country_Region']
#     else:
#         return x['Province_State']


#Adjusting Facttable
dataset = dataset.rename(columns={"ID": "Id", "Province_State": "Region", "Country_Region": "Country" , "DATE": "Date" , "ConfirmedCases": "Confirmed_Cases" , "Fatalities": "Noof_Deaths"})
new_order = [0,2,1,3,4,5]
dataset = dataset[dataset.columns[new_order]]

#EDA
No_of_AffectedNations = dataset.Country.nunique()
Affected_Countries = []
Affected_Countries = dataset.Country.unique()
Affected_Countries = pd.DataFrame(data = Affected_Countries)
#Affected_Countries = Affected_Countries.rename(columns={"0": "Distinct_Nation"})

Affected_Countries_Cases = dataset['Confirmed_Cases'].groupby(dataset['Country']).unique()
Affected_Countries_Cases = Affected_Countries_Cases.to_frame()
#ffected_Countries_Cases = dataset['Noof_Deaths'].groupby(dataset['Country']).unique()

Fatalities_Cross_Nations  = []
Fatalities_Cross_Nations = dataset['Noof_Deaths'].groupby(dataset['Country']).unique()
Affected_Countries_Cases['Fatalities_Cross_Nations'] = Fatalities_Cross_Nations
  
Var_Bin1 = Affected_Countries_Cases["Confirmed_Cases"] 
Length_of_Feature = len(Var_Bin1)
for i in range(0,Length_of_Feature):
    current_rows_arr_len = len(Var_Bin1[i])
    for j in range(0,current_rows_arr_len):
        if j == current_rows_arr_len-1:
            Var_Bin1[i] = Var_Bin1[i][j]

Var_Bin2 = Affected_Countries_Cases['Fatalities_Cross_Nations']
Length_of_Feature_Copy = len(Var_Bin2)
for c in range(0, Length_of_Feature_Copy):
    current_rows_arr_len1 = len(Var_Bin2[c])
    for a in range(0, current_rows_arr_len1):
        if a == current_rows_arr_len1-1:
             Var_Bin2[c] =  Var_Bin2[c][a]
        
Affected_Countries_Cases = Affected_Countries_Cases.reset_index()
Affected_Countries_Cases = Affected_Countries_Cases.sort_values(['Confirmed_Cases'], ascending=[False])
Affected_Countries_Cases = Affected_Countries_Cases.reset_index() 
Affected_Countries_Cases = Affected_Countries_Cases.drop(columns = 'index')                                      

x = Affected_Countries_Cases['Confirmed_Cases']
y = Affected_Countries_Cases['Fatalities_Cross_Nations']
plt.xlabel('Total Confirmed Cases', fontsize=12)
plt.ylabel('Total Calamity', fontsize=12)
plt.title("Covid 19 Widespread", fontsize=25)
plt.scatter(x, y, color='Brown', marker="8")

Top_VictimNations = Affected_Countries_Cases.loc[0:9]
# Top_VictimNations = Top_VictimNations.to_dict()
#Top_VictimNations = Top_VictimNations.set_index('Country').T.to_dict()
My_Dict1 = {'Italy':92472,
            'Spain':73235,
            'Germany':57695,
            'France':37575,
            'Iran':35408,
            'United Kingdom':17089,
            'Switzerland':14076,
            'Netherlands':9762,
            'South Korea':9478,
            'Belguim': 9134}
plt.bar(My_Dict1.keys(), My_Dict1.values(), color='Brown')
plt.xlabel("Countries", fontsize=12)
plt.ylabel("Total Number of Confirmed Cases",fontsize=12)
plt.title("Top 10 Infected Nations", fontsize = 20)
plt.xticks(rotation = 40)
plt.figure(figsize=(12,24))


My_Dict2 = {'Italy':10023,
            'Spain':5982,
            'Germany':433,
            'France':2314,
            'Iran':2517,
            'United Kingdom':1019,
            'Switzerland':264,
            'Netherlands':639,
            'South Korea':144,
            'Belguim': 353}
plt.bar(My_Dict1.keys(), My_Dict1.values(), color='Brown')
plt.xlabel("Countries", fontsize=12)
plt.ylabel("Total Number of Death Cases",fontsize=12)
plt.title("Casulity w.r.t. Countries", fontsize = 20)
plt.xticks(rotation = 40)
plt.figure(figsize=(12,24))

#Country Spefic EDA
#1 Italy
Italy = dataset[dataset.Country == 'Italy']
Italy = Italy.loc[:, ['Date', 'Confirmed_Cases', 'Noof_Deaths']]
Italy = Italy.reset_index()
Italy = Italy.drop(columns = 'index')                                      
sns.set(style='white',)
sns.lineplot(x = "Date", y = "Confirmed_Cases", data=Italy, color= 'Red')
sns.lineplot(x = "Date", y = "Noof_Deaths", data=Italy, color = 'Blue')
plt.title("Covid 19 Knockdown in Italy", fontsize =18)
plt.ylabel('Confirmed_Cases/Casuality')
plt.legend(labels=['No of Ongoing Cases', 'Death Cases'])
plt.show() 

#2 Spain
Spain = dataset[dataset.Country == 'Spain']
Spain = Spain.loc[:, ['Date', 'Confirmed_Cases', 'Noof_Deaths']]
Spain = Spain.reset_index()
Spain = Spain.drop(columns = 'index')                                      
sns.set(style='white',)
sns.lineplot(x = "Date", y = "Confirmed_Cases", data=Spain, color= 'Red')
sns.lineplot(x = "Date", y = "Noof_Deaths", data=Spain, color = 'Blue')
plt.title("Covid 19 Knockdown in Spain", fontsize =18)
plt.ylabel('Confirmed_Cases/Casuality')
plt.legend(labels=['No of Ongoing Cases', 'Death Cases'])
plt.show() 

#3 Germany
Germany = dataset[dataset.Country == 'Germany']
Germany = Germany.loc[:, ['Date', 'Confirmed_Cases', 'Noof_Deaths']]
Germany = Germany.reset_index()
Germany = Germany.drop(columns = 'index')                                      
sns.set(style='white',)
sns.lineplot(x = "Date", y = "Confirmed_Cases", data=Germany, color= 'Red')
sns.lineplot(x = "Date", y = "Noof_Deaths", data=Germany, color = 'Blue')
plt.title("Covid 19 Knockdown in Germany", fontsize =18)
plt.ylabel('Confirmed_Cases/Casuality')
plt.legend(labels=['No of Ongoing Cases', 'Death Cases'])
plt.show() 

#4 Iran
Iran = dataset[dataset.Country == 'Iran']
Iran = Iran.loc[:, ['Date', 'Confirmed_Cases', 'Noof_Deaths']]
Iran = Iran.reset_index()
Iran = Iran.drop(columns = 'index')                                      
sns.set(style='white',)
sns.lineplot(x = "Date", y = "Confirmed_Cases", data=Iran, color= 'Red')
sns.lineplot(x = "Date", y = "Noof_Deaths", data=Iran, color = 'Blue')
plt.title("Covid 19 Knockdown in Iran", fontsize =18)
plt.ylabel('Confirmed_Cases/Casuality')
plt.legend(labels=['No of Ongoing Cases', 'Death Cases'])
plt.show() 

#5 UK
UK = dataset[dataset.Country == 'United Kingdom']
UK = UK.loc[:, ['Date', 'Confirmed_Cases', 'Noof_Deaths']]
UK = UK.reset_index()
UK = UK.drop(columns = 'index')                                      
sns.set(style='white',)
sns.lineplot(x = "Date", y = "Confirmed_Cases", data=UK, color= 'Red')
sns.lineplot(x = "Date", y = "Noof_Deaths", data=UK, color = 'Blue')
plt.title("Covid 19 Knockdown in United Kingdom", fontsize =14)
plt.ylabel('Confirmed_Cases/Casuality')
plt.legend(labels=['No of Ongoing Cases', 'Death Cases'])
plt.show() 

#6 South korea
South_Korea = dataset[dataset.Country == 'Korea, South']
South_Korea = South_Korea.loc[:, ['Date', 'Confirmed_Cases', 'Noof_Deaths']]
South_Korea = South_Korea.reset_index()
South_Korea = South_Korea.drop(columns = 'index')                                      
sns.set(style='white',)
sns.lineplot(x = "Date", y = "Confirmed_Cases", data=South_Korea, color= 'Red')
sns.lineplot(x = "Date", y = "Noof_Deaths", data=South_Korea, color = 'Blue')
plt.title("Covid 19 Knockdown in South Korea", fontsize =18)
plt.ylabel('Confirmed_Cases/Casuality')
plt.legend(labels=['No of Ongoing Cases', 'Death Cases'])
plt.show() 

#7 China
China = dataset[dataset.Country == 'China']
China = China.loc[:, ['Date', 'Confirmed_Cases', 'Noof_Deaths']]
China = China.reset_index()
China = China.drop(columns = 'index')                                      
sns.set(style='white',)
sns.lineplot(x = "Date", y = "Confirmed_Cases", data=China, color= 'Red')
sns.lineplot(x = "Date", y = "Noof_Deaths", data=China, color = 'Blue')
plt.title("Covid 19 Knockdown in China", fontsize =18)
plt.ylabel('Confirmed_Cases/Casuality')
plt.legend(labels=['No of Ongoing Cases', 'Death Cases'])
plt.show() 

import geopandas as gpd
Map = pd.DataFrame()
Map = pd.concat([Affected_Countries_Cases, pd.DataFrame(columns = [ 'Country', 'Confirmed_Cases'])])
Map = Map.drop(columns = ['Fatalities_Cross_Nations'])
world_data = gpd.read_file(r'D:\Kaggle_Clashes\Covid-19\World_Map.shp')

for items in Map['Country'].tolist():
    world_data_list = world_data['NAME'].tolist()
    if items in world_data_list:
        pass
    else:
        print(items + ' is not in the world_data list')
world_data.replace('Korea, Republic of', 'South Korea', inplace = True)
world_data.replace('Iran (Islamic Republic of)', 'Iran', inplace = True)
world_data.rename(columns = {'NAME': 'Country'}, inplace = True)
Map_Viz = world_data.merge(Map, on = 'Country')
Map_Viz.plot(column='Confirmed_Cases', legend=False, cmap='OrRd')
plt.title('Effect of Covid 19 Across Globe', fontsize=15)
plt.xlabel([])
plt.ylabel([])
plt.xticks([])
plt.yticks([])


#Predicting Confirmed Cases w.r.t Date
dataset1 = pd.read_csv('test.csv')

X1_train = dataset.Date
X1_train = X1_train.to_frame()
X1_train =  pd.to_datetime(X1_train.Date)
X1_train = X1_train.dt.weekday #Monday is 0, Sunday is 6
X1_train = X1_train.to_frame()
X1_train = X1_train.rename(columns={"Date": "Day"})

Y1_train = dataset.Confirmed_Cases
Y1_train = Y1_train.to_frame()

X1_test = dataset1.Date
X1_test = X1_test.to_frame()
X1_test =  pd.to_datetime(X1_test.Date)
X1_test = X1_test.dt.weekday 
X1_test = X1_test.to_frame()
X1_test = X1_test.rename(columns={"Date": "Day"})

from sklearn.naive_bayes import GaussianNB
new_model = GaussianNB()
new_model.fit(X1_train, Y1_train)
X1_prediction = new_model.predict(X1_test)
X1_prediction = pd.DataFrame(X1_prediction)
#X1_prediction = X1_prediction.rename(columns={"0": "Confirmed_Cases"})
X1_prediction.columns=["Confirmed_Cases"]


#Predicting Fatalities w.r.t Date and Confirmed Cases
X2_train = dataset.drop(['Id', 'Country', 'Region', 'Noof_Deaths'], axis=1)
X2_train['Date'] =  pd.to_datetime(X2_train.Date)
X2_train['Date'] = X2_train['Date'].dt.weekday 
X2_train = X2_train.rename(columns={"Date": "Day"})

Y2_train = dataset.Noof_Deaths

X2_test = dataset1.Date
X2_test = X2_test.to_frame()
X2_test =  pd.to_datetime(X2_test.Date)
X2_test = X2_test.dt.weekday 
X2_test = X2_test.to_frame()
X2_test['Confirmed_Cases'] = X1_prediction['Confirmed_Cases']

from sklearn.tree import DecisionTreeRegressor
new_model1 = DecisionTreeRegressor(random_state = 27)
new_model1.fit(X2_train, Y2_train)
X2_prediction = new_model1.predict(X2_test)
X2_prediction = pd.DataFrame(X2_prediction)
X2_prediction = X2_prediction.astype(int)
X2_prediction.columns=['Fatalities']

# from sklearn.metrics import mean_absolute_error
# print("Accuracy in train set : ", new_model.score(X1_train, Y1_train))
# print("Accuracy in train set : ", new_model1.score(X2_train, Y2_train))

Submission = dataset1.ForecastId
Submission = Submission.to_frame()
Submission['ConfirmedCases'] = X1_prediction['Confirmed_Cases']
Submission['Fatalities'] = X2_prediction['Fatalities']

Submission.to_csv('Submission_Covid19.csv', index=True)

