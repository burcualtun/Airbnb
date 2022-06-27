import streamlit as st
import datetime as dt
from PIL import Image
import numpy as np
from sklearn.preprocessing import  RobustScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

#from helpers.Data_prep import *
#from helpers.Models import *

st.set_page_config(layout='wide',initial_sidebar_state ='expanded',page_title="MIULBNB!",page_icon="🏡")

#SIDEBAR
st.sidebar.header('USER INPUT FEATURES')
def user_input_features():
    Accommodates=st.sidebar.number_input(label='Host Capacity',min_value=1,max_value=8,value=3)
    guests_included = st.sidebar.number_input(label='Number of Guest', min_value=0, max_value=5, value=2)
    ExtraPeople = st.sidebar.slider('How much Euros Each Extra Person?', 0, 100, 0)
    Bedrooms = st.sidebar.number_input(label='Bedroom Count', min_value=1, max_value=10, value=2)
    Bathrooms=st.sidebar.number_input(label='Bathrooms Count', min_value=0, max_value=3, value=1)
    RoomType = st.sidebar.radio('Room Type Selection', ('Entire home/apt', 'Private room','Shared room'))
    PropertyType = st.sidebar.selectbox('Property Type Selection', ('Apartment', 'House', 'Bed & Breakfast','Boat','Loft','Other','Cabin','Camper/RV',                                                            'Villa'))
    Neighbourhood = st.sidebar.selectbox('Neighbourhood Selection', ('Centrum-West', 'De Baarsjes - Oud-West','Centrum-Oost','De Pijp - Rivierenbuurt',
    'Westerpark','Zuid','Oud-Oost','Bos en Lommer','Oud-Noord','Watergraafsmeer','Slotervaart','IJburg - Zeeburgereiland','Buitenveldert - Zuidas',
    'Noord-West','Geuzenveld - Slotermeer','Noord-Oost','Osdorp','De Aker - Nieuw Sloten','Bijlmer-Centrum','Bijlmer-Oost','Gaasperdam - Driemond'))
    #ReviewScoreCheckin = st.sidebar.number_input('Checkin Review Score', min_value=0, max_value=10, value=0)
    #ReviewScoreLocation = st.sidebar.number_input('Location Review Score',min_value=0, max_value=10, value=0)
    #ReviewScoreCommunication = st.sidebar.number_input('Communication Review Score', min_value=0, max_value=30, value=0)
    #ReviewScoreAccuracy = st.sidebar.number_input('Accuracy Review Score', min_value=0, max_value=10, value=0)

    data = {'Accommodates': [Accommodates],
            'guests_included': [guests_included],
            'ExtraPeople': [ExtraPeople],
            'Bedrooms': [Bedrooms],
            'Bathrooms': [Bathrooms],
            'RoomType': [RoomType],
            'PropertyType': [PropertyType],
            'Neighbourhood': [Neighbourhood]
            #'ReviewScoreCheckin': [ReviewScoreCheckin],
            #'ReviewScoreLocation': [ReviewScoreLocation],
            #'ReviewScoreCommunication': [ReviewScoreCommunication],
            #'ReviewScoreAccuracy': [ReviewScoreAccuracy]
            }

    features = pd.DataFrame(data,index=[0])

    return features

#Sayfaya Resim ve Başlık Ekle
image = Image.open('WhatsApp Image 2022-06-17 at 21.07.16.jpeg')
st.image(image,width=800)
st.header('MIUULBNB PRICE PREDICTION APP')

#Seçilen Değerleri DF yapıp göster
input_df = user_input_features()
st.header('User Choices')
st.write(input_df)

#Girilen değerler için df oluştur
#New Person By Area Oluştur
df=pd.read_csv('Unit_1_Project_Dataset.csv')
df['zipcode'] = df['zipcode'].str[0:4]
neighbourhood_cleansed_mode = df.groupby(['neighbourhood_cleansed'])['zipcode'].agg(pd.Series.mode).reset_index()
input_df['zip_code']=neighbourhood_cleansed_mode[neighbourhood_cleansed_mode['neighbourhood_cleansed']==input_df['Neighbourhood'][0]]['zipcode'].values

DataFrame_detay = pd.read_csv("Amsterdam_nufus.csv", sep=";", encoding='unicode_escape')
df['zipcode'] = df['zipcode'].astype(str)
DataFrame_detay['zipcode'] = DataFrame_detay['zipcode'].astype(str)

input_df['Population'] = (DataFrame_detay[DataFrame_detay['zipcode']==input_df['zip_code'][0]]['Population']).values
input_df['Area'] = (DataFrame_detay[DataFrame_detay['zipcode']==input_df['zip_code'][0]]['Area']).values
input_df['NEW_person_By_Area']=input_df['Population']/input_df['Area']
input_df.drop(['zip_code','Population','Area'],axis=1,inplace=True)

#NEW_DISTRICT oluştur.
input_df.loc[input_df['Neighbourhood'] == 'Centrum-West', 'NEW_DISTRICT'] = 'Center'
input_df.loc[input_df['Neighbourhood'] == 'De Baarsjes - Oud-West', 'NEW_DISTRICT'] = 'West'
input_df.loc[input_df['Neighbourhood'] == 'Centrum-Oost', 'NEW_DISTRICT'] = 'Center'
input_df.loc[input_df['Neighbourhood'] == 'De Pijp - Rivierenbuurt', 'NEW_DISTRICT'] = 'Zuid'
input_df.loc[input_df['Neighbourhood'] == 'Westerpark', 'NEW_DISTRICT'] = 'West'
input_df.loc[input_df['Neighbourhood'] == 'Zuid', 'NEW_DISTRICT'] = 'Zuid'
input_df.loc[input_df['Neighbourhood'] == 'Oud-Oost', 'NEW_DISTRICT'] = 'Oost'
input_df.loc[input_df['Neighbourhood'] == 'Bos en Lommer', 'NEW_DISTRICT'] = 'West'
input_df.loc[input_df['Neighbourhood'] == 'Oostelijk Havengebied - Indische Buurt', 'NEW_DISTRICT'] = 'Oost'
input_df.loc[input_df['Neighbourhood'] == 'Oud-Noord', 'NEW_DISTRICT'] = 'Noord'
input_df.loc[input_df['Neighbourhood'] == 'Watergraafsmeer', 'NEW_DISTRICT'] = 'Oost'
input_df.loc[input_df['Neighbourhood'] == 'Slotervaart', 'NEW_DISTRICT'] = 'Nieuw-West'
input_df.loc[input_df['Neighbourhood'] == 'Geuzenveld - Slotermeer', 'NEW_DISTRICT'] = 'Nieuw-West'
input_df.loc[input_df['Neighbourhood'] == 'De Aker - Nieuw Sloten', 'NEW_DISTRICT'] = 'Nieuw-West'
input_df.loc[input_df['Neighbourhood'] == 'Osdorp', 'NEW_DISTRICT'] = 'Nieuw-West'
input_df.loc[input_df['Neighbourhood'] == 'IJburg - Zeeburgereiland', 'NEW_DISTRICT'] = 'Oost'
input_df.loc[input_df['Neighbourhood'] == 'Buitenveldert - Zuidas', 'NEW_DISTRICT'] = 'Zuid'
input_df.loc[input_df['Neighbourhood'] == 'Noord-West', 'NEW_DISTRICT'] = 'Noord'
input_df.loc[input_df['Neighbourhood'] == 'Noord-Oost', 'NEW_DISTRICT'] = 'Noord'
input_df.loc[input_df['Neighbourhood'] == 'Bijlmer-Centrum', 'NEW_DISTRICT'] = 'Zuidoost'
input_df.loc[input_df['Neighbourhood'] == 'Bijlmer-Oost', 'NEW_DISTRICT'] = 'Zuidoost'
input_df.loc[input_df['Neighbourhood'] == 'Gaasperdam - Driemond', 'NEW_DISTRICT'] = 'Zuidoost'

input_df.drop(['Neighbourhood'],axis=1,inplace=True)

#Model kur
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols, num_but_cat
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

#Feature Engineering
def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
def FeatureEngineering (DataFrame):
    #Öncelikle dublike kayıtları sil.
    # Dublike id'leri siliyoruz.
    x = DataFrame.groupby("id").agg({"id": "count"})
    x.columns = ['Cnt']
    x.reset_index(inplace=True)
    dublicated_id = x[x.Cnt > 1]['id']

    lst = []
    for col in dublicated_id:
        del_index = DataFrame[DataFrame['id'] == col].index.min()
        lst.append(del_index)

    DataFrame = DataFrame[~DataFrame.index.isin(lst)]

    # 1
    # host_since_year+host_since_anniversary ile yeni tarih türetilecek.ve bu değişkenler silinecek
    # df.drop('market',axis=1,inplace=True)
    DataFrame['marker'] = DataFrame['host_since_anniversary'].str.find('/')
    DataFrame.loc[DataFrame['marker'] == 1, 'host_since_anniversary'] = '0' + DataFrame[DataFrame['marker'] == 1]['host_since_anniversary']

    DataFrame['date_len'] = DataFrame['host_since_anniversary'].str.len()
    DataFrame['right_digit'] = DataFrame['host_since_anniversary'].str[-1:]
    DataFrame.loc[DataFrame['date_len'] == 4, 'host_since_anniversary'] = DataFrame[DataFrame['date_len'] == 4]['host_since_anniversary'].str[
                                                            0:3] + '0' + DataFrame['right_digit']
    DataFrame.drop(['date_len', 'right_digit', 'marker'], axis=1, inplace=True)

    DataFrame['host_since_date'] = pd.to_datetime(DataFrame['host_since_year'].astype(str) + '/' + DataFrame['host_since_anniversary'])
    DataFrame.drop(['host_since_year', 'host_since_anniversary'], axis=1, inplace=True)

    # Analizin yapıldığın gün 25.06.2015 olsun
    today_date = dt.datetime(2015, 6, 25)
    DataFrame['host_since_days'] = (today_date - DataFrame['host_since_date']).dt.days
    DataFrame.drop(['host_since_date'], axis=1, inplace=True)

    # host_since_days'i kategoriğe çevir
    night_bins = [81, 360, 588, 807, 1080, 2509]
    night_labels = ["81_360", "361_588", "589_807", "808_1080", "1081_2509"]

    DataFrame["NEW_Host_Start_Day_CAT"] = pd.cut(DataFrame["host_since_days"], bins=night_bins, labels=night_labels)
    DataFrame.drop(['host_since_days'], axis=1, inplace=True)

    #2
    #Neighbourhood bilgisine göre Amsterdam'ın 7 bölgesini ekleyebiliriz.
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Centrum-West', 'NEW_DISTRICT'] = 'Center'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'De Baarsjes - Oud-West', 'NEW_DISTRICT'] = 'West'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Centrum-Oost', 'NEW_DISTRICT'] = 'Center'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'De Pijp - Rivierenbuurt', 'NEW_DISTRICT'] = 'Zuid'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Westerpark', 'NEW_DISTRICT'] = 'West'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Zuid', 'NEW_DISTRICT'] = 'Zuid'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Oud-Oost', 'NEW_DISTRICT'] = 'Oost'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Bos en Lommer', 'NEW_DISTRICT'] = 'West'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Oostelijk Havengebied - Indische Buurt', 'NEW_DISTRICT'] = 'Oost'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Oud-Noord', 'NEW_DISTRICT'] = 'Noord'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Watergraafsmeer', 'NEW_DISTRICT'] = 'Oost'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Slotervaart', 'NEW_DISTRICT'] = 'Nieuw-West'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Geuzenveld - Slotermeer', 'NEW_DISTRICT'] = 'Nieuw-West'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'De Aker - Nieuw Sloten', 'NEW_DISTRICT'] = 'Nieuw-West'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Osdorp', 'NEW_DISTRICT'] = 'Nieuw-West'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'IJburg - Zeeburgereiland', 'NEW_DISTRICT'] = 'Oost'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Buitenveldert - Zuidas', 'NEW_DISTRICT'] = 'Zuid'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Noord-West', 'NEW_DISTRICT'] = 'Noord'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Noord-Oost', 'NEW_DISTRICT'] = 'Noord'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Bijlmer-Centrum', 'NEW_DISTRICT'] = 'Zuidoost'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Bijlmer-Oost', 'NEW_DISTRICT'] = 'Zuidoost'
    DataFrame.loc[DataFrame['neighbourhood_cleansed'] == 'Gaasperdam - Driemond', 'NEW_DISTRICT'] = 'Zuidoost'

    #3
    #DataFrame['NEW_Acc_Multiply_Beds'] = DataFrame['accommodates'] * DataFrame['beds']

    #4
    DataFrame['NEW_Bed_Divide_Person'] = DataFrame['guests_included'] / DataFrame['beds']

    #5
    DataFrame['zipcode'] = DataFrame['zipcode'].str[0:4]
    neighbourhood_cleansed_mode = DataFrame.groupby(['neighbourhood_cleansed'])['zipcode'].agg(pd.Series.mode).reset_index()
    DataFrame = pd.merge(DataFrame, neighbourhood_cleansed_mode, how="left", on=["neighbourhood_cleansed"])
    DataFrame.loc[DataFrame['zipcode_x'].isnull(), 'zipcode_x'] = DataFrame[DataFrame['zipcode_x'].isnull()]['zipcode_y']
    DataFrame.drop('zipcode_y', inplace=True, axis=1)
    DataFrame.rename({'zipcode_x': 'zipcode'}, axis=1, inplace=True)

    #İnternetten bulunan veri ile merge edilir.
    DataFrame_detay = pd.read_csv("Amsterdam_nufus.csv", sep=";", encoding='unicode_escape')

    DataFrame['zipcode'] = DataFrame['zipcode'].astype(str)
    DataFrame_detay['zipcode'] = DataFrame_detay['zipcode'].astype(str)

    DataFrame = pd.merge(DataFrame, DataFrame_detay, how="left", on=["zipcode"])
    DataFrame['NEW_person_By_Area'] = DataFrame['Population'] / DataFrame['Area']
    DataFrame.drop(['Population','Area'],axis=1,inplace=True)

    return DataFrame
df=FeatureEngineering(df)

#missing value
def MissingValueHandle(dataframe):
    review_columns = dataframe.loc[:, (dataframe.columns.str.contains('review'))].columns
    dataframe[review_columns] = dataframe[review_columns].fillna(0)
    dataframe['host_response_time'] = dataframe['host_response_time'].fillna('None')
    dataframe['host_response_rate'] = dataframe['host_response_rate'].fillna(0)
    # State:%99'u north Holland. Useless column. Boşları North Holland yapabiliriz.
    dataframe['state'].fillna('North Holland', inplace=True)
    dataframe.dropna(inplace=True)
    return dataframe
df=MissingValueHandle(df)

#Handle Outlier
df.drop(df[df['price']>300].index,inplace=True)
for col in num_cols:
    replace_with_thresholds(df, col)

def LOFOutlierDetection(DataFrame,n_neighbors,num_cols):
    df_ = DataFrame[num_cols]

    # LOF Scores:
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    clf.fit_predict(df_)
    df_scores = clf.negative_outlier_factor_

    # LOF Visualization:
    scores = pd.DataFrame(np.sort(df_scores))
    scores.plot(stacked=True, xlim=[0, 10], style='.-')
    plt.show(block=True)
    return df_scores
df_scores=LOFOutlierDetection(df,20,num_cols)
th = np.sort(df_scores)[0:10][2]
df.drop(df[df_scores < th].index, inplace=True)


model_columns=['NEW_person_By_Area','NEW_DISTRICT','room_type','property_type','accommodates','guests_included','extra_people'
    ,'bedrooms','bathrooms'
    #,'review_scores_checkin','review_scores_location','review_scores_accuracy','review_scores_communication'
               ]

#input data ile csv'yi birleştir.
input_df.columns=['accommodates','guests_included','extra_people','bedrooms','bathrooms','room_type','property_type',
                 # 'review_scores_checkin','review_scores_location','review_scores_communication','review_scores_accuracy',
                  'NEW_person_By_Area', 'NEW_DISTRICT'
                  ]
All_data=pd.concat([input_df,df[model_columns]],axis=0)

cat_cols=['room_type','property_type','NEW_DISTRICT'
  #  ,'review_scores_checkin','review_scores_location','review_scores_communication','review_scores_accuracy'
          ]
num_cols = [col for col in input_df.columns if col not in ['price','room_type','property_type','NEW_DISTRICT',
                                                           #'review_scores_checkin','review_scores_location',
                                                       # 'review_scores_communication','review_scores_accuracy'
] ]
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
df_encode=one_hot_encoder(All_data,cat_cols)

def RobustScaling(dataframe, col_name):
    rs = RobustScaler()
    dataframe[col_name] = rs.fit_transform(dataframe[col_name])
    return dataframe
df_encode[num_cols]=RobustScaling(df_encode[num_cols],num_cols)

df_tahmin=df_encode[:1]



#Modeli ekle
df_model=pd.read_csv('Unit_1_Project_Dataset.csv')

##########################################################
#feature engineering
###################################################
df_model=FeatureEngineering(df_model)

#missing value
df_model=MissingValueHandle(df_model)

#Outlier detection
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df_model)
num_cols.remove("review_scores_cleanliness")
cat_cols.append("review_scores_cleanliness")


df_model.drop(df_model[df_model['price']>300].index,inplace=True)
for col in num_cols:
    replace_with_thresholds(df_model, col)

df_scores=LOFOutlierDetection(df_model,20,num_cols)
th = np.sort(df_scores)[0:10][2]
df_model.drop(df_model[df_scores < th].index, inplace=True)

#Kullanılacak kolonlar
model_columns=['NEW_person_By_Area','NEW_DISTRICT','room_type','property_type','accommodates','guests_included','extra_people'
    ,'bedrooms','bathrooms'#,'review_scores_checkin','review_scores_location','review_scores_accuracy','review_scores_communication'
                ,'price']

df_model=df_model[model_columns]
cat_cols=['room_type','property_type','NEW_DISTRICT'
    #,'review_scores_checkin','review_scores_location'
    #,'review_scores_communication','review_scores_accuracy'
          ]
num_cols = [col for col in model_columns if col not in ['room_type','property_type','NEW_DISTRICT',
            #'review_scores_checkin','review_scores_location','review_scores_communication','review_scores_accuracy'
                                                        ] ]
#Encoding
df_encode=one_hot_encoder(df_model,cat_cols)

#Scaling
num_cols = [col for col in num_cols if "price" not in col]
df_encode[num_cols]=RobustScaling(df_encode[num_cols],num_cols)

X = df_encode.drop('price', axis=1)
y = df_encode['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=112)
#alphas= 10**np.linspace(10,-2,100)*0.5
#lasso_cv = RidgeCV(alphas = alphas, cv = 10)
#lasso_cv_model = lasso_cv.fit(X,y)

ls = Ridge(alpha=9.369087114301934)
lasso_tuned_model = ls.fit(X_train,y_train)
prediction=lasso_tuned_model.predict(df_tahmin)

#import shap
#explainer=shap.TreeExplainer(load_model)
#shap_values=explainer.shap_values(df_encode)

st.header('Price Prediction Result')
prediction

#Map ekle

#plotting a map with the above defined points
#data=pd.read_csv('Unit_1_Project_Dataset.csv')

#st.header("Price Distribution in Amsterdam Neibourhood?")
# plot the slider that selects number of person died
#prices = st.slider("Price of Lots", int(data["price"].min()), int(data["price"].max()))
#bedds = st.slider("How Many Beds", int(data["beds"].min()), int(data["beds"].max()))
#st.map(data.query("price <= @prices & beds<=@bedds")[["latitude", "longitude"]].dropna(how ="any"))



