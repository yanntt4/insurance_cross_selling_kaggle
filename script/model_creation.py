# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:12:32 2023

@author: ythiriet
"""


# Global importation
import sys
import math
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
import optuna
import random
# import lightgbm #LGBMRegressor
import catboost #CatBoostRegressor
from sklearn.ensemble import StackingRegressor, StackingClassifier
from scikeras.wrappers import KerasRegressor, KerasClassifier
from sklearn.preprocessing import OneHotEncoder
import joblib
import numbers

from sklearn.impute import KNNImputer

import tensorflow as tf
from sklearn.metrics import matthews_corrcoef
import optuna

# Personal library
sys.path.append("./class")
from Data_modelling import Data_modelling
from Data_modelling_random_forest import random_forest
from Data_modelling_gradient_boosting import gradient_boosting
from Data_modelling_neural_network import neural_network
from Data_modelling_xgboosting import xgboosting
from Data_modelling_catboosting import catboosting



# Class containing all parameters
class Parameters():
    def __init__(self):
        self.CLEAR_MODE = True
        
        self.NAME_DATA_PREDICT = "Response"
        self.GENERIC_NAME_DATA_PREDICT = "Assurance subscription ?" # for plot

        self.SWITCH_REMOVING_DATA = True
        self.LIST_DATA_DROP = ["id", "Policy_Sales_Channel"]
        self.SWITCH_DATA_REDUCTION = False
        self.SWITCH_DATA_NOT_ENOUGHT = False
        self.NB_DATA_NOT_ENOUGHT = 1500
        self.SWITCH_ABERRANT_IDENTICAL_DATA = True
        self.SWITCH_RELATION_DATA = False
        self.ARRAY_RELATION_DATA = np.array([["Height", 2],["Age", 2]], dtype = object)
        self.SWITCH_ENCODE_DATA_PREDICT = False
        self.ARRAY_DATA_ENCODE_PREDICT = np.array([[self.NAME_DATA_PREDICT,"p",1],[self.NAME_DATA_PREDICT,"e",0]], dtype = object)
        self.SWITCH_ENCODE_DATA = True
        self.SWITCH_ENCODE_DATA_ONEHOT = False
        self.LIST_DATA_ENCODE_ONEHOT = ["Gender", "Vehicle_Age", "Vehicle_Damage"]
        self.ARRAY_DATA_ENCODE_REPLACEMENT = np.zeros(3, dtype = object)
        self.ARRAY_DATA_ENCODE_REPLACEMENT[0] = np.array(
            [["Gender","Female",1,"FEMME"],["Gender","Male",0,"HOMME"]], dtype = object)
        self.ARRAY_DATA_ENCODE_REPLACEMENT[1] = np.array(
            [["Vehicle_Age","< 1 Year",0,"Neuf"],["Vehicle_Age","1-2 Year",1,"Occasion/Neuf"],["Vehicle_Age","> 2 Years",1,"Occasion"]], dtype = object)
        self.ARRAY_DATA_ENCODE_REPLACEMENT[2] = np.array(
            [["Vehicle_Damage","Yes",1,"OUI"],["Vehicle_Damage","No",0,"NON"]], dtype = object)

        self.SWITCH_PLOT_DATA = False
        self.SWITCH_EQUILIBRATE_DATA = False
        self.SWITCH_SMOTEN_DATA = False
        self.SWITCH_REPLACING_NAN = False
        self.SWITCH_SAMPLE_DATA = True
        self.FRACTION_SAMPLE_DATA = 0.01

        self.RF_MODEL = False
        self.RF_MODEL_OPTI = True
        self.RF_MODEL_TRIAL = 25
        
        self.GB_MODEL = False
        self.GB_MODEL_OPTI = True
        self.GB_MODEL_TRIAL = 25

        self.NN_MODEL = False
        self.NN_MODEL_OPTI = True
        self.NN_MODEL_TRIAL = 5

        self.XG_MODEL = False
        self.XG_MODEL_OPTI = False
        self.XG_MODEL_TRIAL = 25
        
        self.CB_MODEL = True
        self.CB_MODEL_OPTI = False
        self.CB_MODEL_TRIAL = 10

        self.MULTI_CLASSIFICATION = False

        self.N_SPLIT = 5
        self.k_folds = KFold(n_splits=self.N_SPLIT)


    # Determining if multi-classification
    def multi_classification_analysis(self, UNIQUE_PREDICT_VALUE):

        if UNIQUE_PREDICT_VALUE.shape[0] > 2:
            self.MULTI_CLASSIFICATION = True
    
    
    def regression_analysis(self, TRAIN_DATAFRAME):
        if isinstance(TRAIN_DATAFRAME[self.NAME_DATA_PREDICT][0], numbers.Number):
            self.REGRESSION = True
        else:
            self.REGRESSION = False
    
    def saving_array_replacement(self):
        joblib.dump(self.ARRAY_DATA_ENCODE_REPLACEMENT, "./data_replacement/array_data_encode_replacement.joblib")
        

class Data_Preparation():
    def __init__(self):
        self.TRAIN_DATAFRAME = []
        self.TEST_DATAFRAME = []
        self.TRAIN_STATS = []
        self.UNIQUE_PREDICT_VALUE = []
        self.TRAIN_CORRELATION = []
        self.DUPLICATE_LINE = []

        self.ARRAY_REPLACEMENT_ALL = np.zeros([0], dtype = object)
        self.INDEX_REPLACEMENT_ALL = np.zeros([0], dtype = object)


    def data_import(self, NAME_DATA_PREDICT):

        self.TRAIN_DATAFRAME = pd.read_csv("./data/train.csv")
        self.TEST_DATAFRAME = pd.read_csv("./data/test.csv")
        self.TRAIN_STATS = self.TRAIN_DATAFRAME.describe()


    def data_predict_description(self, NAME_DATA_PREDICT):
        self.UNIQUE_PREDICT_VALUE = self.TRAIN_DATAFRAME.groupby(NAME_DATA_PREDICT)[NAME_DATA_PREDICT].count()
        
        # Printing first values
        print(self.TRAIN_DATAFRAME.head())


    def data_encoding_replacement(self, ARRAY_REPLACEMENT, NAN_VALUES = False):
        
        for i_encoding, DataFrame in enumerate([self.TRAIN_DATAFRAME, self.TEST_DATAFRAME]):
    
            # Replacement
            for j in range(ARRAY_REPLACEMENT.shape[0]):
                for k in range(ARRAY_REPLACEMENT[j].shape[0]):
                    DataFrame[ARRAY_REPLACEMENT[j][k][0]] = DataFrame[ARRAY_REPLACEMENT[j][k][0]].replace(
                        ARRAY_REPLACEMENT[j][k][1], int(ARRAY_REPLACEMENT[j][k][2]))
                
            # Replacing nan values
            if NAN_VALUES:
                DataFrame[ARRAY_REPLACEMENT[j][0][0]] = DataFrame[ARRAY_REPLACEMENT[j][0][0]].fillna(0)

            # Recording the replacement
            if i_encoding == 0:
                self.TRAIN_DATAFRAME = DataFrame
            else:
                self.TEST_DATAFRAME = DataFrame


    def data_encoding_replacement_important(self, COLUMN_NAME):

        # Init
        self.ARRAY_REPLACEMENT_ALL = np.append(
            self.ARRAY_REPLACEMENT_ALL, np.zeros([1], dtype = object), axis = 0)
        self.INDEX_REPLACEMENT_ALL = np.append(
            self.INDEX_REPLACEMENT_ALL, np.zeros([1], dtype = object), axis = 0)

        DF_TRAIN_TEST = pd.concat([Global_Data.TRAIN_DATAFRAME, Global_Data.TEST_DATAFRAME],
                                  ignore_index = True)
        UNIQUE_DF_TRAIN_TEST = DF_TRAIN_TEST.groupby(COLUMN_NAME)[COLUMN_NAME].count()
        ARRAY_REPLACEMENT = pd.DataFrame(UNIQUE_DF_TRAIN_TEST.index).to_numpy()
        INDEX_REPLACEMENT = pd.DataFrame(UNIQUE_DF_TRAIN_TEST.index).index.tolist()

        for i_encoding, DataFrame in enumerate([self.TRAIN_DATAFRAME, self.TEST_DATAFRAME]):

            # Replacement
            for j in range(ARRAY_REPLACEMENT.shape[0]):
                DataFrame[COLUMN_NAME] = DataFrame[COLUMN_NAME].replace(
                    ARRAY_REPLACEMENT[j], INDEX_REPLACEMENT[j])

            # Recording the replacement
            if i_encoding == 0:
                self.TRAIN_DATAFRAME[COLUMN_NAME] = DataFrame[COLUMN_NAME]
            else:
                self.TEST_DATAFRAME[COLUMN_NAME] = DataFrame[COLUMN_NAME]

        # Recording the replacement
        self.ARRAY_REPLACEMENT_ALL[-1] = ARRAY_REPLACEMENT
        self.INDEX_REPLACEMENT_ALL[-1] = INDEX_REPLACEMENT


    def data_encoding_replacement_predict(self, ARRAY_REPLACEMENT):
        for j in range(ARRAY_REPLACEMENT.shape[0]):
            self.TRAIN_DATAFRAME[ARRAY_REPLACEMENT[j,0]] = self.TRAIN_DATAFRAME[ARRAY_REPLACEMENT[j,0]].replace(
                ARRAY_REPLACEMENT[j,1],ARRAY_REPLACEMENT[j,2])


    def data_encoding_onehot(self, NAME_DATA_ENCODE):
        Enc = OneHotEncoder(handle_unknown='ignore')
        DATA_ENCODE_TRAIN = self.TRAIN_DATAFRAME.loc[:,[NAME_DATA_ENCODE]]
        DATA_ENCODE_TEST = self.TEST_DATAFRAME.loc[:,[NAME_DATA_ENCODE]]
        DATA_ENCODE_NAME = pd.Series(NAME_DATA_ENCODE + DATA_ENCODE_TRAIN.groupby(NAME_DATA_ENCODE)[NAME_DATA_ENCODE].count().index)
        DATA_ENCODE_NAME = DATA_ENCODE_NAME.replace(["<",">"]," ", regex=True)
        Enc.fit(DATA_ENCODE_TRAIN)

        DATA_ENCODE_TRAIN = Enc.transform(DATA_ENCODE_TRAIN).toarray()
        DATA_ENCODE_TRAIN = pd.DataFrame(DATA_ENCODE_TRAIN, columns = DATA_ENCODE_NAME)
        DATA_ENCODE_TRAIN = DATA_ENCODE_TRAIN.set_index(self.TRAIN_DATAFRAME.index)

        self.TRAIN_DATAFRAME = self.TRAIN_DATAFRAME.drop(columns = NAME_DATA_ENCODE)
        self.TRAIN_DATAFRAME = pd.concat([self.TRAIN_DATAFRAME, DATA_ENCODE_TRAIN], axis = 1)

        DATA_ENCODE_TEST = Enc.transform(DATA_ENCODE_TEST).toarray()
        DATA_ENCODE_TEST = pd.DataFrame(DATA_ENCODE_TEST, columns = DATA_ENCODE_NAME)
        DATA_ENCODE_TEST = DATA_ENCODE_TEST.set_index(self.TEST_DATAFRAME.index)

        self.TEST_DATAFRAME = self.TEST_DATAFRAME.drop(columns = NAME_DATA_ENCODE)
        self.TEST_DATAFRAME = pd.concat([self.TEST_DATAFRAME, DATA_ENCODE_TEST], axis = 1)
    
    
    def encode_data_error_removal(self, ARRAY_REPLACEMENT):
        for ARRAY in ARRAY_REPLACEMENT:
            Global_Data.TRAIN_DATAFRAME[ARRAY[0][0]] = pd.to_numeric(
                Global_Data.TRAIN_DATAFRAME[ARRAY[0][0]],errors="coerce", downcast = 'integer')
    
    
    def data_format_removal(self, ARRAY_REPLACEMENT, Type = str, Len = 2):
        for ARRAY in ARRAY_REPLACEMENT:
            Global_Data.TRAIN_DATAFRAME = Global_Data.TRAIN_DATAFRAME.loc[(
                Global_Data.TRAIN_DATAFRAME[ARRAY[0][0]].astype(Type).str.len() < Len)]
        

    def data_drop(self, Name_data_drop):
        self.TRAIN_DATAFRAME = self.TRAIN_DATAFRAME.drop([Name_data_drop],axis=1)
        self.TEST_DATAFRAME = self.TEST_DATAFRAME.drop([Name_data_drop],axis=1)


    def data_pow(self, Name_Data_Duplicate, Number_Duplicate):
        self.TRAIN_DATAFRAME[Name_Data_Duplicate] = (
            self.TRAIN_DATAFRAME[Name_Data_Duplicate].pow(Number_Duplicate))
        self.TEST_DATAFRAME[Name_Data_Duplicate] = (
            self.TEST_DATAFRAME[Name_Data_Duplicate].pow(Number_Duplicate))


    def data_duplicate_removal(self, NAME_DATA_PREDICT, Column_Drop = ""):

        if len(Column_Drop) == 0:
            Duplicated_Data_All = self.TRAIN_DATAFRAME.drop(NAME_DATA_PREDICT, axis = 1).duplicated()
        else:
            Duplicated_Data_All = self.TRAIN_DATAFRAME.drop([Column_Drop, NAME_DATA_PREDICT],axis = 1).duplicated()
        self.DUPLICATE_LINE = Duplicated_Data_All.loc[Duplicated_Data_All == True]
        self.TRAIN_DATAFRAME = self.TRAIN_DATAFRAME.drop(self.DUPLICATE_LINE.index)
        
        # Information to the user
        print(f"{self.DUPLICATE_LINE.shape[0]} has been removed because of duplicates")
        plot.pause(3)


    def remove_low_data(self, NB_DATA_NOT_ENOUGHT, NAME_DATA_NOT_ENOUGHT, LIST_NAME_DATA_REMOVE_MULTIPLE = []):

        # Searching for data with low values
        TRAIN_GROUP_VALUE = self.TRAIN_DATAFRAME.groupby(NAME_DATA_NOT_ENOUGHT)[NAME_DATA_NOT_ENOUGHT].count().sort_values(ascending = False)

        # Adding values only inside NAME DATA REMOVE MULTIPLE
        for NAME_DATA_REMOVE_MULTIPLE in LIST_NAME_DATA_REMOVE_MULTIPLE:
            TRAIN_GROUP_VALUE_OTHER = self.TRAIN_DATAFRAME.groupby(NAME_DATA_REMOVE_MULTIPLE)[NAME_DATA_REMOVE_MULTIPLE].count().index

        for VALUE_OTHER in TRAIN_GROUP_VALUE_OTHER:
            if np.sum(VALUE_OTHER == np.array(TRAIN_GROUP_VALUE.index)) == 0:
                TRAIN_GROUP_VALUE = pd.concat([TRAIN_GROUP_VALUE, pd.Series(0, index = [VALUE_OTHER])])

        # Searching for values to drop following number of elements
        REMOVE_TRAIN_GROUP_VALUE = TRAIN_GROUP_VALUE.drop(TRAIN_GROUP_VALUE[TRAIN_GROUP_VALUE > NB_DATA_NOT_ENOUGHT].index)

        # Removing data inside train and test dataframe
        for DATA_REMOVE in REMOVE_TRAIN_GROUP_VALUE.index:
            self.TRAIN_DATAFRAME = self.TRAIN_DATAFRAME.drop(self.TRAIN_DATAFRAME[self.TRAIN_DATAFRAME[NAME_DATA_NOT_ENOUGHT] == DATA_REMOVE].index)
            self.TEST_DATAFRAME = self.TEST_DATAFRAME.drop(self.TEST_DATAFRAME[self.TEST_DATAFRAME[NAME_DATA_NOT_ENOUGHT] == DATA_REMOVE].index)

            for NAME_DATA_REMOVE_MULTIPLE in LIST_NAME_DATA_REMOVE_MULTIPLE:
                self.TRAIN_DATAFRAME = self.TRAIN_DATAFRAME.drop(self.TRAIN_DATAFRAME[self.TRAIN_DATAFRAME[NAME_DATA_REMOVE_MULTIPLE] == DATA_REMOVE].index)
                self.TEST_DATAFRAME = self.TEST_DATAFRAME.drop(self.TEST_DATAFRAME[self.TEST_DATAFRAME[NAME_DATA_REMOVE_MULTIPLE] == DATA_REMOVE].index)

        # Reseting index
        self.TRAIN_DATAFRAME = self.TRAIN_DATAFRAME.reset_index(drop = True)
        self.TEST_DATAFRAME = self.TEST_DATAFRAME.reset_index(drop = True)


    def oversampling(self, NAME_DATA_PREDICT, NB_DATA_NOT_ENOUGHT, Name_Data_Oversample = ""):

        self.UNIQUE_PREDICT_VALUE = self.TRAIN_DATAFRAME.groupby(NAME_DATA_PREDICT)[NAME_DATA_PREDICT].count()
        Max_Nb_Data = np.amax(self.UNIQUE_PREDICT_VALUE.to_numpy())

        if len(Name_Data_Oversample) > 1:
            Global_Table_Train_Equilibrate = self.UNIQUE_PREDICT_VALUE.loc[(
                self.UNIQUE_PREDICT_VALUE.index == "Overweight_Level_II")]
        else:
            # Global_Table_Train_Equilibrate = self.UNIQUE_PREDICT_VALUE.loc[
            #     self.UNIQUE_PREDICT_VALUE > NB_DATA_NOT_ENOUGHT]
            Global_Table_Train_Equilibrate = self.UNIQUE_PREDICT_VALUE.loc[self.UNIQUE_PREDICT_VALUE < Max_Nb_Data]

        for i in tqdm(range(Global_Table_Train_Equilibrate.shape[0])):
            Matrix_To_Add = np.zeros(
                [0, self.TRAIN_DATAFRAME.shape[1]],
                dtype=object)
            DF_Reference = self.TRAIN_DATAFRAME.loc[self.TRAIN_DATAFRAME[NAME_DATA_PREDICT] == pd.DataFrame(
                Global_Table_Train_Equilibrate.index).iloc[i][0]]
            for j in tqdm(range(Max_Nb_Data - Global_Table_Train_Equilibrate.iloc[i])):
                Matrix_To_Add = np.append(
                    Matrix_To_Add,
                    np.zeros([1, self.TRAIN_DATAFRAME.shape[1]],
                              dtype=object),
                    axis=0)

                Matrix_To_Add[-1, :] = DF_Reference.iloc[
                    random.randint(0, DF_Reference.shape[0] - 1), :].to_numpy()

            DataFrame_To_Add = pd.DataFrame(
                Matrix_To_Add,
                columns=self.TRAIN_DATAFRAME.columns)

            self.TRAIN_DATAFRAME = pd.concat(
                [self.TRAIN_DATAFRAME, DataFrame_To_Add],
                ignore_index=True)


    def data_sample(self, SAMPLE_FRACTION):

        self.TRAIN_DATAFRAME = self.TRAIN_DATAFRAME.sample(
            frac = SAMPLE_FRACTION, replace = False, random_state = 42)


    def nan_replacing(self, COLUMN_NAMES):
        

        # Creating a column indicating missing value
        for COLUMN in COLUMN_NAMES:
            self.TRAIN_DATAFRAME[f"{COLUMN} Missing"] = self.TRAIN_DATAFRAME[f"{COLUMN}"].isnull()
        
        # Replacing missing values with nearest neightboor
        COLUMNS = self.TRAIN_DATAFRAME.columns
        imputer = KNNImputer(n_neighbors=20, weights="uniform")
        TRAIN_ARRAY = imputer.fit_transform(self.TRAIN_DATAFRAME)
        
        # Turning numpy array into dataframe
        self.TRAIN_DATAFRAME = pd.DataFrame(TRAIN_ARRAY, columns = COLUMNS)
    
    
    def saving_data_names(self):
        joblib.dump(self.TEST_DATAFRAME.columns, "./data_replacement/data_names.joblib")
        

class Data_Plot():
    def __init__(self):
        self.BOX_PLOT_DATA_PREDICT = ""
        self.BOX_PLOT_DATA_AVAILABLE = ""
        self.CORRELATION_PLOT = ""
        self.TRAIN_DATAFRAME = []
        self.TRAIN_CORRELATION = []
        self.UNIQUE_PREDICT_VALUE = []


    def box_plot_data_predict_plot(self, GENERIC_NAME_DATA_PREDICT):

        # Init
        fig, self.BOX_PLOT_DATA_PREDICT = plot.subplots(2)
        plot.suptitle(f"Data count following {GENERIC_NAME_DATA_PREDICT}",
                      fontsize = 25,
                      color = "gold",
                      fontweight = "bold")

        # Horizontal bars for each possibilities
        self.BOX_PLOT_DATA_PREDICT[0].barh(
            y = self.UNIQUE_PREDICT_VALUE.index,
            width=self.UNIQUE_PREDICT_VALUE,
            height=0.03,
            label=self.UNIQUE_PREDICT_VALUE.index)

        # Cumulative horizontal bars
        Cumulative_Value = 0
        for i in range(self.UNIQUE_PREDICT_VALUE.shape[0]):
            self.BOX_PLOT_DATA_PREDICT[1].barh(
                y=1,
                width=self.UNIQUE_PREDICT_VALUE.iloc[i],
                left = Cumulative_Value)
            self.BOX_PLOT_DATA_PREDICT[1].text(
                x = Cumulative_Value + 100,
                y = 0.25,
                s = self.UNIQUE_PREDICT_VALUE.index[i])
            Cumulative_Value += self.UNIQUE_PREDICT_VALUE.iloc[i]
        self.BOX_PLOT_DATA_PREDICT[1].set_ylim(0, 2)
        self.BOX_PLOT_DATA_PREDICT[1].legend(
            self.UNIQUE_PREDICT_VALUE.index.to_numpy(),
            ncol=int(self.UNIQUE_PREDICT_VALUE.shape[0]/2),
            fontsize=6)


    def plot_data_repartition(self):

        # Init
        NB_LINE = math.ceil((self.TRAIN_DATAFRAME.shape[1] - 1)/3)
        NB_COLUMN = math.ceil((self.TRAIN_DATAFRAME.shape[1] - 1)/3)

        # Box Plot for all data
        fig, self.BOX_PLOT_DATA_AVAILABLE = plot.subplots(NB_LINE, NB_COLUMN)
        plot.suptitle("Boxplot for all data into the TRAIN dataset",
                      fontsize = 25,
                      color = "chartreuse",
                      fontweight = "bold")

        for i in range(NB_LINE):
            for j in range(NB_COLUMN):
                if i*NB_COLUMN + j < self.TRAIN_DATAFRAME.shape[1]:
                    try:
                        self.BOX_PLOT_DATA_AVAILABLE[i, j].boxplot(
                            self.TRAIN_DATAFRAME.iloc[:, [i*NB_COLUMN + j]])
                        self.BOX_PLOT_DATA_AVAILABLE[i, j].set_title(
                            self.TRAIN_DATAFRAME.iloc[:, i*NB_COLUMN + j].name,
                            fontweight = "bold",
                            fontsize = 15)
                    except:
                        continue
    
    
    def plot_data_hist(self):

        # Init
        NB_LINE = math.ceil((self.TRAIN_DATAFRAME.shape[1] - 1)/3)
        NB_COLUMN = math.ceil((self.TRAIN_DATAFRAME.shape[1] - 1)/3)

        # Box Plot for all data
        fig, self.BOX_PLOT_DATA_AVAILABLE = plot.subplots(NB_LINE, NB_COLUMN)
        plot.suptitle("Boxplot for all data into the TRAIN dataset",
                      fontsize = 25,
                      color = "chartreuse",
                      fontweight = "bold")

        for i in range(NB_LINE):
            for j in range(NB_COLUMN):
                if (i*NB_COLUMN + j < self.TRAIN_DATAFRAME.shape[1]):
                    try:
                        self.BOX_PLOT_DATA_AVAILABLE[i, j].hist(
                            self.TRAIN_DATAFRAME.iloc[:, [i*NB_COLUMN + j]],
                            bins = 100)
                        self.BOX_PLOT_DATA_AVAILABLE[i, j].set_title(
                            self.TRAIN_DATAFRAME.iloc[:, i*NB_COLUMN + j].name,
                            fontweight = "bold",
                            fontsize = 15)
                    except:
                        continue


    def plot_data_relation(self, NAME_DATA_X, NAME_DATA_Y):

        plot.figure()
        plot.scatter(self.TRAIN_DATAFRAME[NAME_DATA_X], self.TRAIN_DATAFRAME[NAME_DATA_Y])
        plot.suptitle(
            f"Relation between {NAME_DATA_X} and {NAME_DATA_Y} variables",
            fontsize = 25,
            color = "darkorchid",
            fontweight = "bold")


    def CORRELATION_PLOT_Plot(self):

        fig2, self.CORRELATION_PLOT = plot.subplots()
        im = self.CORRELATION_PLOT.imshow(
            self.TRAIN_CORRELATION,
            vmin=-1,
            vmax=1,
            cmap="bwr")
        self.CORRELATION_PLOT.figure.colorbar(im, ax=self.CORRELATION_PLOT)
        self.CORRELATION_PLOT.set_xticks(np.linspace(
            0, self.TRAIN_DATAFRAME.shape[1] - 1, self.TRAIN_DATAFRAME.shape[1]))
        self.CORRELATION_PLOT.set_xticklabels(np.array(self.TRAIN_DATAFRAME.columns, dtype = str),
                                              rotation = 45)
        self.CORRELATION_PLOT.set_yticks(np.linspace(
            0, self.TRAIN_DATAFRAME.shape[1] - 1, self.TRAIN_DATAFRAME.shape[1]))
        self.CORRELATION_PLOT.set_yticklabels(np.array(self.TRAIN_DATAFRAME.columns, dtype = str))



# -- ////////// --
# -- ////////// --
# -- ////////// --





# Init for global parameters
Global_Parameters = Parameters()

if Global_Parameters.CLEAR_MODE:

    # Removing data
    from IPython import get_ipython
    get_ipython().magic('reset -sf')

    # Closing all figures
    plot.close("all")


Global_Data = Data_Preparation()
Global_Data.data_import(Global_Parameters.NAME_DATA_PREDICT)
Global_Parameters.regression_analysis(Global_Data.TRAIN_DATAFRAME)
Global_Parameters.REGRESSION = False


# Droping some columns
if Global_Parameters.SWITCH_REMOVING_DATA:
    for name_drop in Global_Parameters.LIST_DATA_DROP:
        Global_Data.data_drop(name_drop)


# Removing variable with too low data
if Global_Parameters.SWITCH_DATA_REDUCTION:
    Global_Data.remove_low_data(
        Global_Parameters.NB_DATA_NOT_ENOUGHT, "Origin",
        LIST_NAME_DATA_REMOVE_MULTIPLE = ["Dest"])

# Data description
Global_Data.data_predict_description(Global_Parameters.NAME_DATA_PREDICT)

# Multi classification identification
Global_Parameters.multi_classification_analysis(Global_Data.UNIQUE_PREDICT_VALUE)


# Sample Data
if Global_Parameters.SWITCH_SAMPLE_DATA:
    Global_Data.data_sample(Global_Parameters.FRACTION_SAMPLE_DATA)


# Encoding data for entry variables
# sys.exit()
if Global_Parameters.SWITCH_ENCODE_DATA:
    if Global_Parameters.SWITCH_ENCODE_DATA_ONEHOT:
        for NAME_DATA_ENCODE in Global_Parameters.LIST_DATA_ENCODE_ONEHOT:
            Global_Data.data_encoding_onehot(NAME_DATA_ENCODE)

    else:
        # # Removing data with incorrect format before encoding
        # Global_Data.data_format_removal(Global_Parameters.ARRAY_DATA_ENCODE_REPLACEMENT)
        
        # Encoding
        Global_Data.data_encoding_replacement(Global_Parameters.ARRAY_DATA_ENCODE_REPLACEMENT, True)
    
        # Removing error data after encoding
        Global_Data.encode_data_error_removal(Global_Parameters.ARRAY_DATA_ENCODE_REPLACEMENT)
        Global_Data.TRAIN_DATAFRAME = Global_Data.TRAIN_DATAFRAME.dropna()


# Encoding data for predict variable
if Global_Parameters.SWITCH_ENCODE_DATA_PREDICT:
    Global_Data.data_encoding_replacement_predict(Global_Parameters.ARRAY_DATA_ENCODE_PREDICT)


# Searching for and removing aberrant/identical values
if Global_Parameters.SWITCH_ABERRANT_IDENTICAL_DATA:
    Global_Data.data_duplicate_removal(Global_Parameters.NAME_DATA_PREDICT)


# Oversampling to equilibrate data
if (Global_Parameters.SWITCH_EQUILIBRATE_DATA and Global_Parameters.SWITCH_SMOTEN_DATA == False):
    Global_Data.oversampling(Global_Parameters.NAME_DATA_PREDICT, Global_Parameters.NB_DATA_NOT_ENOUGHT)


# Searching for repartition on data to predict
if Global_Parameters.SWITCH_PLOT_DATA:

    Global_Data_Plot = Data_Plot()
    Global_Data_Plot.TRAIN_DATAFRAME = Global_Data.TRAIN_DATAFRAME
    Global_Data_Plot.UNIQUE_PREDICT_VALUE = Global_Data.UNIQUE_PREDICT_VALUE
    Global_Data_Plot.box_plot_data_predict_plot(Global_Parameters.GENERIC_NAME_DATA_PREDICT)
    Global_Data_Plot.plot_data_repartition()
    Global_Data_Plot.plot_data_hist()
    plot.pause(1)
    # Global_Data_Plot.plot_data_relation("Height", "Gender")
    plot.pause(1)
    Global_Data.TRAIN_CORRELATION = Global_Data.TRAIN_DATAFRAME.iloc[
        :,:Global_Data.TRAIN_DATAFRAME.shape[1] - 1].corr()
    Global_Data_Plot.TRAIN_CORRELATION = Global_Data.TRAIN_CORRELATION
    Global_Data_Plot.CORRELATION_PLOT_Plot()


# Modifying linear relation between data
if Global_Parameters.SWITCH_RELATION_DATA:
    for i in range(Global_Parameters.List_Relation_Data.shape[0]):
        Global_Data.data_pow(Global_Parameters.List_Relation_Data[i,0],
                             Global_Parameters.List_Relation_Data[i,1])


# Replacing Nan values
if Global_Parameters.SWITCH_REPLACING_NAN:
    Global_Data.nan_replacing(["BsmtQual", "BsmtCond", "BsmtExposure"])

# Generic Data Model
Data_Model = Data_modelling()
Data_Model.splitting_data(Global_Data.TRAIN_DATAFRAME,
                          Global_Parameters.NAME_DATA_PREDICT,
                          Global_Parameters.MULTI_CLASSIFICATION,
                          Global_Parameters.REGRESSION)
if (Global_Parameters.SWITCH_SMOTEN_DATA and Global_Parameters.SWITCH_EQUILIBRATE_DATA):
    Data_Model.smoten_sampling()


#
# Random Forest

if Global_Parameters.RF_MODEL:
    DATA_MODEL_RF = random_forest(Data_Model, Global_Parameters, Global_Data)


#
# Gradient Boosting

if Global_Parameters.GB_MODEL:
    DATA_MODEL_GB = gradient_boosting(Data_Model, Global_Parameters, Global_Data)
    

#
# Neural Network

if Global_Parameters.NN_MODEL:
    DATA_MODEL_NN = neural_network(Data_Model, Global_Parameters, Global_Data)


#
# XGBoosting

if Global_Parameters.XG_MODEL:
    DATA_MODEL_XG = xgboosting(Data_Model, Global_Parameters, Global_Data)


#
# Catboosting
if Global_Parameters.CB_MODEL:
    DATA_MODEL_CB = catboosting(Data_Model, Global_Parameters, Global_Data)



# Saving model and information
Global_Parameters.saving_array_replacement()
Global_Data.saving_data_names()

if Global_Parameters.RF_MODEL:
    with open('./models/rf_model.sav', 'wb') as f:
        joblib.dump(DATA_MODEL_RF.MODEL, f)
elif Global_Parameters.NN_MODEL:
    with open('./models/nn_model.sav', 'wb') as f:
        joblib.dump(DATA_MODEL_NN.MODEL, f)
elif Global_Parameters.GB_MODEL:
    with open('./models/gb_model.sav', 'wb') as f:
        joblib.dump(DATA_MODEL_GB.MODEL, f)
elif Global_Parameters.XG_MODEL:
    with open('./models/xg_model.sav', 'wb') as f:
        joblib.dump(DATA_MODEL_XG.MODEL, f)

# # Kaggle competition
# for NAME in ["cap-shape","cap-color","does-bruise-or-bleed","gill-color","stem-color","has-ring","ring-type","habitat"]:
#     Global_Data.TEST_DATAFRAME[NAME] = pd.to_numeric(Global_Data.TEST_DATAFRAME[NAME], errors = "coerce").fillna(0)

# A = pd.DataFrame(DATA_MODEL_XG.MODEL.predict(Global_Data.TEST_DATAFRAME), columns = ["class"])
# # A = pd.DataFrame(np.argmax(DATA_MODEL_NN.MODEL.predict(Global_Data.TEST_DATAFRAME), axis = 1), columns = ["class"])
# A = A.replace(0,"e")
# A = A.replace(1,"p")
# Global_Data.data_import(Global_Parameters.NAME_DATA_PREDICT)
# A.index = Global_Data.TEST_DATAFRAME.id
# A.to_csv("kaggle_compet.csv", index_label = "id")


# Model Stacking

# Store all the base models in a list
estimators = [
    ('NN', KerasClassifier(DATA_MODEL_NN.MODEL)),
    ('XG', DATA_MODEL_XG.MODEL),
]

# Create the stacked model with the base models and Elastic Net as the meta-model
stack = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier())
stack.fit(Data_Model.X_train, Data_Model.Y_train)

# If using XG model, need to deactivate early stopping and eval set (and, therefore,
# plot associated with)