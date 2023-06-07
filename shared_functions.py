#!/usr/bin/env python
# coding: utf-8

# # Projet 7 - Implementation of a scoring model
# # Notebook - Shared Functions

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Projet-7---Implementation-of-a-scoring-model" data-toc-modified-id="Projet-7---Implementation-of-a-scoring-model-1">Projet 7 - Implementation of a scoring model</a></span></li><li><span><a href="#Notebook---Shared-Functions" data-toc-modified-id="Notebook---Shared-Functions-2">Notebook - Shared Functions</a></span></li><li><span><a href="#I)-Importations-and-global-settings" data-toc-modified-id="I)-Importations-and-global-settings-3">I) Importations and global settings</a></span><ul class="toc-item"><li><span><a href="#1)-Importation-of-required-libraries" data-toc-modified-id="1)-Importation-of-required-libraries-3.1">1) Importation of required libraries</a></span></li><li><span><a href="#2)-Settings-of-global-graphics-parameters" data-toc-modified-id="2)-Settings-of-global-graphics-parameters-3.2">2) Settings of global graphics parameters</a></span></li><li><span><a href="#3)-Global-files'-path" data-toc-modified-id="3)-Global-files'-path-3.3">3) Global files' path</a></span></li></ul></li><li><span><a href="#II)-Functions" data-toc-modified-id="II)-Functions-4">II) Functions</a></span><ul class="toc-item"><li><span><a href="#1)-Basics" data-toc-modified-id="1)-Basics-4.1">1) Basics</a></span></li><li><span><a href="#2)-Dataframes-optimization" data-toc-modified-id="2)-Dataframes-optimization-4.2">2) Dataframes optimization</a></span></li><li><span><a href="#3)-Model-fitting-and-predictions" data-toc-modified-id="3)-Model-fitting-and-predictions-4.3">3) Model fitting and predictions</a></span></li><li><span><a href="#4)-Optimization-of-the-probability-threshold" data-toc-modified-id="4)-Optimization-of-the-probability-threshold-4.4">4) Optimization of the probability threshold</a></span></li><li><span><a href="#5)-Theoritical-number-of-FP-(<=>-Reverse-of-the-job-score-in-absolute-values)" data-toc-modified-id="5)-Theoritical-number-of-FP-(<=>-Reverse-of-the-job-score-in-absolute-values)-4.5">5) Theoritical number of FP (&lt;=&gt; Reverse of the job score in absolute values)</a></span></li><li><span><a href="#6)-AUROC" data-toc-modified-id="6)-AUROC-4.6">6) AUROC</a></span></li><li><span><a href="#7)-F-bêta-score" data-toc-modified-id="7)-F-bêta-score-4.7">7) F-bêta score</a></span></li><li><span><a href="#8)-Confusion-matrix" data-toc-modified-id="8)-Confusion-matrix-4.8">8) Confusion matrix</a></span></li><li><span><a href="#9)-Classification-indicators" data-toc-modified-id="9)-Classification-indicators-4.9">9) Classification indicators</a></span></li><li><span><a href="#10)-Job-score" data-toc-modified-id="10)-Job-score-4.10">10) Job score</a></span></li><li><span><a href="#11)-MLFlow-Tracker" data-toc-modified-id="11)-MLFlow-Tracker-4.11">11) MLFlow Tracker</a></span></li><li><span><a href="#12)-Table-to-store-all-models'-relevant-values-along-notebooks" data-toc-modified-id="12)-Table-to-store-all-models'-relevant-values-along-notebooks-4.12">12) Table to store all models' relevant values along notebooks</a></span></li><li><span><a href="#13)-SHAP" data-toc-modified-id="13)-SHAP-4.13">13) SHAP</a></span></li><li><span><a href="#14)-Any-python-object-serialization-to-string-and-deserialization" data-toc-modified-id="14)-Any-python-object-serialization-to-string-and-deserialization-4.14">14) Any python object serialization to string and deserialization</a></span></li></ul></li></ul></div>

# # I) Importations and global settings

# ## 1) Importation of required libraries

# In[1]:


### File management ###

# Files' path.
import os.path

# Save and load files.
import csv
import pickle
import base64 # Allow to seriliaze and deserialize any object as a string with pickle.


### Models tracking and records ###

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

### Data manipulations ###

import numpy as np
from numpy import set_printoptions # Saving full data when exporting to csv format.
import pandas as pd


### Date & time ###

# Time measurment and datetime management
import datetime as dt
from time import time


### Warnings removal ###

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


### Data visualizations ###

from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns


### Additional common libraries ###

from numpy import argmax, argmin
import math
from random import sample as py_rd_sp # Python random sampling.

# Those allow to transform the shap values from their logodd format to odd.
import copy
from scipy.special import expit # Opposed of logit.


### sklearn tools & libraries ###

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_predict, cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, fbeta_score, confusion_matrix
from sklearn.metrics import make_scorer # Allow to make a sklearn custom scorer (For the custom job score).


### Imbalanced data management ###

from imblearn.pipeline import Pipeline # NB: imbalearn.pipeline.Pipeline allows to properly deal the SMOTE on the train set and avoid the validation/test sets.
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTENC # NB: SMOTENC can manage categorial features while SMOTE cannot.


### Features interpretability ###

import shap


# ## 2) Settings of global graphics parameters

# In[2]:


### Set default figure parameters for the whole notebook ###

# Default parameters for matplotlib's figures.
#plt.rcParams['figure.figsize'] = [6,6]
#plt.rcParams['figure.dpi'] = 200
#plt.rcParams['axes.spines.right'] = False
#plt.rcParams['axes.spines.top'] = False
#plt.rcParams['xtick.bottom'] = True
#plt.rcParams['ytick.left'] = True

# Default parameters of seaborn's figures.
#sns.set_style('white') # NB: Needs to be above sns.set_theme to properly attend custom_params.
#custom_params = {'axes.spines.right': False, 'axes.spines.top': False}
#sns.set_theme(style='ticks', palette='deep', rc=custom_params) # All seaborn and matplolib figures will display with the seaborn's configurations..


# ## 3) Global files' path

# In[3]:


# Global file paths.
#EXPORTS_DIR_PATH = 'Exports'
EXPORTS_MODELS_DIR_PATH = r'Exports\Models\Tried'
IMPORTS_DIR_PATH = r'Exports\Preprocessed_data'
MLFLOW_EXPORTS_MODEL_DIR_PATH = r'Exports\Models\Tried\MLFlow'

CSV_MODELS_FILE = 'models_info.csv'
PKL_MODELS_FILE = 'models_info.pkl'
#JSON_MODELS_FILE = 'models_info.json'
#DATASETS_DIR_PATH = r'D:\0Partage\MP-P2PNet\MP-Sync\MP-Sync_Pro\Info\OC_DS\Projet 7\Datasets' #os.path.join('D:', '0Partage', 'MP-P2PNet', 'MP-Sync', 'MP-Sync_Pro', 'Info', 'OC_DS', 'Projet 7', 'Datasets')


# # II) Functions

# ## 1) Basics

# In[4]:


def display_EZ (x, max_rows = 100, max_cols = 100, max_colwidth = 100):
    
    """
    Description
    -----------
    Allows to display pandas dataframes with the number of rows and columns whished in an easy manner.
    
    Parameters
    ----------
    df: pandas.DataFrame()
        Dataframe to display.
    max_rows: int
        Maximum number of rows to display.
    max_cols: int
        Maximum number of columns to display.
    max_colwidth: int
        Maximum width of each column.
        
    """
    
    with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_cols, 'display.max_colwidth', max_colwidth):
        display(x)


# In[5]:


def df_to_csv_full (df):
    
    """ 
    Long numpy.array() stored within a cell of the csv file are truncated and stringify
    loosing most of the information and breaking the object.
    This small function changes the numpy.array() displaying limit to truncate the object for its full exportation and
    avoid data loss.
        
    """
    
    # Set the numpy array number of items cutting threshold to a very high number and avoid the cut.
    set_printoptions(threshold=1e100, linewidth=1e100)
    
    # Save the df to a csv file.
    df.to_csv(os.path.join(EXPORTS_MODELS_DIR_PATH, CSV_MODELS_FILE))
    
    # Reset the numpy array number of items cutting threshold to default.
    set_printoptions(threshold=100, linewidth=25)


# ## 2) Dataframes optimization

# In[6]:


def reduce_memory (df):
    
    """ Reduce memory usage of a dataframe by setting data types. """

    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Initial df memory usage is {:.2f} MB for {} columns'
          .format(start_mem, len(df.columns)))

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            cmin = df[col].min()
            cmax = df[col].max()
            if str(col_type)[:3] in ['Int', 'int']:
                # Can use unsigned int here too
                if cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif cmin > np.iinfo(np.int64).min and cmax < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if cmin > np.finfo(np.float16).min and cmax < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
    end_mem = df.memory_usage().sum() / 1024 ** 2
    memory_reduction = 100 * (start_mem - end_mem) / start_mem
    print('Final memory usage is: {:.2f} MB - decreased by {:.1f}%'.format(end_mem, memory_reduction))
    
    return df


# In[7]:


def find_int_cols (df):
    
    """
    Find and change the column dtype as int for columns with only int values.
    This allows to spot all numerical categorical columns.
    
    """
    
    for col in df.columns:
        for item in df[col]:
            
            # Test all items within the column to check if they all are integers.
            if item == int(item):
                if df[col].dtype != "int64":
                    df[col] = df[col].astype("int64")
                    
            # If only one of the items tested is not an int, the test is stopped and it switch to the next column.
            else:
                if df[col].dtype != "float64":
                    df[col] = df[col].astype("float64") 
                break
                
    return df


# ## 3) Model fitting and predictions

# In[8]:


def model_fit_predict (model, X, y, cv, X_test=None):
    
    """ Fit the model and get its predictions probabilities over a defined validation set or cross validation. """
    
    
    # Fit and predict probabilities by cross validation (=> No need to fit it before and does not make much sense).
    if cv != 0:
        t0 = time()
        yhat = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
        process_time = time() - t0

    # Fit and predict (no cv).
    else:
        t0 = time()
        model.fit(X, y)
        yhat = model.predict_proba(X_test)
        process_time = time() - t0
    
    return yhat, process_time


# In[9]:


def get_y_pred_list (y_pred_proba_P, l_proba_thrs = np.linspace(0, 1, num=101)):
    
    """ Classify all customers of the sample for each probability threshold. """
    
    l_y_pred = []
    for proba_thr in l_proba_thrs:

        y_pred = []
        for proba in y_pred_proba_P:
            if proba > proba_thr:
                y_pred.append(1)
            else:
                y_pred.append(0)
        
        l_y_pred.append(y_pred)
                
    return l_y_pred


# In[10]:


def get_tp_fp_fn_tn_lists (y_true, l_y_pred):

    """ Get the confusion matrix values (binary classification). """
    
    l_tp = []
    l_fp = []
    l_fn = []
    l_tn = []
    for y_pred in l_y_pred:
        #cm = confusion_matrix(y_true, y_pred)
        #l_tn.append(cm[0][0])
        #l_fp.append(cm[0][1])
        #l_fn.append(cm[1][0])
        #l_tp.append(cm[1][1])
        #display(y_true[y_true == 1])
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        l_tp.append(tp)
        l_fp.append(fp)
        l_fn.append(fn)
        l_tn.append(tn)

    
    return l_tp, l_fp, l_fn, l_tn


# ## 4) Optimization of the probability threshold

# In[11]:


def opt_proba_thr (np_tp, np_fp, np_fn, np_tn, l_proba_thrs, fn_cost_coeff = 10):     
        
    """ Find the best probability threshold for the passed data. """
        
    ### Calculation of the best probability threshold.
    
    # Method 1 (Less accurate at low number of thresholds tried): Get the index of the closest FP/FN ratio of 1.
    #J = np_fp / (fn_cost_coeff * np_fn)
    #j_opt = 0
    #for idx, j in enumerate(J):
    #    if (j < 1 and j > j_opt) or (j > 1 and j < j_opt):
    #        j_opt = j
    #        J_opt_idx = idx
    #print("Corresponding optimal ratio found:", J[J_opt_idx])
    
    # Method 2 (More accurate at low number of thresholds tried):
    # NB: As it had been seen within the global EDA notebook, there is 1 FN for 10 FP.
    #     Our manager suggested that we should consider that 1 FN cost ~10 times more then 1 FP.
    #     => No need to add a coefficient in front of FN to add more weight since it is already
    #        taken into account within the classification balancement for the 2nd method.
    #        => fn_cost_coeff = 10 --> 1
    
    tpr = np_tp / (np_tp + fn_cost_coeff/fn_cost_coeff * np_fn)
    fpr = np_fp / (np_fp + np_tn)
    J = tpr - fpr
    J_opt_idx = argmax(J)
    
    # Method 3:
    #np_fp_hypothesis = np_fp + fn_cost_coeff * np_fn
    #J_opt_idx = argmin(np_fp_hypothesis)
    
    # Get the optimal probability threshold.
    best_thr = l_proba_thrs[J_opt_idx]

    # Print scores.
    #print('Best Threshold: %.3f' % best_thr)
    #print('Number of FP =', np_fp[J_opt_idx])
    #print('Number of FN =', np_fn[J_opt_idx], '~ FP =', 10 * np_fn[J_opt_idx])
    #print('Equivalent number of FP =', np_fp_hypothesis[J_opt_idx])
    
    return best_thr, J_opt_idx


# In[12]:


def figure_density (y_true, y_pred_proba_P, best_thr, return_fig = False):
    
    """ Find and show the optimal probability threshold on a figure. """
    
    # Find the best threshold value graphically.
    y_true_P = y_true[y_true == 1]
    y_true_N = y_true[y_true == 0]
    
    if return_fig:
        # Create a matplotlib figure object.
        fig, ax = plt.subplots(1)
    
    # Plot the probability density approximation of the TN.
    if return_fig:
        #plt.hist(y_pred_proba_P[y_true_N.index], bins=100, density=True)
        kde_N = sns.kdeplot(y_pred_proba_P[y_true_N.index], ax=ax, fill=True, alpha=0.5, edgecolor='k') #multiple="stack"

        # Plot the probability density approximation of the FN.
        #plt.hist(y_pred_proba_P[y_true_P.index], bins=100, density=True)
        sns.kdeplot(y_pred_proba_P[y_true_P.index], ax=ax, fill=True, alpha=0.5, edgecolor='k')
    else:
        kde_N = sns.kdeplot(y_pred_proba_P[y_true_N.index], fill=True, alpha=0.5, edgecolor='k') #multiple="stack"

        # Plot the probability density approximation of the FN.
        #plt.hist(y_pred_proba_P[y_true_P.index], bins=100, density=True)
        sns.kdeplot(y_pred_proba_P[y_true_P.index], fill=True, alpha=0.5, edgecolor='k')

    # Plot a line at the best threshold found.
    plt.vlines(best_thr, ymin=0, ymax=max(kde_N.get_yticks()), colors='k', linestyles='--')
    
    # Set other figures' parameters.
    plt.title("Distribution of the probability a customer default")
    plt.xlabel("Probability thresholds")
    plt.legend(["Regular customers", "Default customers"])
    
    if return_fig:
        return fig


# ## 5) Theoritical number of FP (<=> Reverse of the job score in absolute values)

# In[13]:


def figure_sum_fp_coeff_fn (np_fp, np_fn, l_proba_thrs, best_thr, fn_cost_coeff = 10,
                            return_fig = False, thr_line_vis = True, model_label = None):
    
    """ 
    According to the coefficients set (FN = 10 x FP) calculate the theorical corresponding total number of
    false predictions for each probability thresholds passed and draw the graph.
        => Equivalent to the inversed gain curve.
    
    """
       
    # Apply the cost hypothesis and convert FN to its supposed corresponding number of FP.
    np_fp_hypothesis = np_fp + fn_cost_coeff * np_fn
          
    if return_fig:
        
        # Create a matplotlib figure object.
        fig, ax = plt.subplots(1)
        
        # Plot the corresponding curve.
        ax.plot(l_proba_thrs, np_fp_hypothesis, label=model_label)
     
        # Plot a line at the best threshold found.
        ax.vlines(best_thr, ymin=0, ymax=max(np_fp_hypothesis), colors='k', linestyles='--', visible=thr_line_vis)
            
    else:
        
        # Plot the corresponding curve.
        plt.plot(l_proba_thrs, np_fp_hypothesis, label=model_label)
     
        # Plot a line at the best threshold found.
        plt.vlines(best_thr, ymin=0, ymax=max(np_fp_hypothesis), colors='k', linestyles='--', visible=thr_line_vis)
        
    print(model_label)
    
    # Set other figures' parameters.
    plt.title("Total number of theorical FP function of probability thresholds")
    plt.xlabel("Probability thresholds")
    plt.ylabel("Total false positives")
    plt.legend()
    
    if return_fig:
        return fig


# ## 6) AUROC

# In[14]:


def figure_roc (y_true, l_yhats, l_model_labels, return_fig = False):
    
    """ Draw the ROC graph and show its corresponding AUC value in the legend. """
    
    if return_fig:
        # Create a matplotlib figure object.
        fig, ax = plt.subplots(1)
    
    #idx = 0
    for idx in range(len(l_model_labels)):
        model_label = l_model_labels[idx]
        yhat = l_yhats[idx]
        
        # Calculate inputs for the roc curve.
        fpr, tpr, thresholds = roc_curve(y_true, yhat)
        
        # Calculate the corresponding AUC.
        auroc = roc_auc_score(y_true, yhat)
    
        # Plot the roc curves.
        if return_fig:
            ax.plot(fpr, tpr, marker='.', markersize=2, label=model_label + " (AUC = %.3f)" % auroc)
        else:
            plt.plot(fpr, tpr, marker='.', markersize=2, label=model_label + " (AUC = %.3f)" % auroc)
        
        # Iterate the index value for the next loop.
        idx += 1
 
    # Plot the no skill roc curve (the diagonal line).
    if return_fig:
        ax.plot([0, 1], [0, 1], linestyle='--', label='No skill (AUC = 0.5)', color='k', alpha=0.75)
    else:
        plt.plot([0, 1], [0, 1], linestyle='--', label='No skill (AUC = 0.5)', color='k', alpha=0.75)
    
    # Set axis labels and the title.
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    
    # Show the legend.
    plt.legend()
    
    if return_fig:
        return fig


# ## 7) F-bêta score

# In[15]:


def get_fbeta_score (l_proba_thrs, l_fbeta, beta, best_thr, best_thr_idx):
    
    """ Get the F-Bêta score for the best threshold found. """
    
    # Get the optimal F-bêta score.
    print('F-Bêta score of the optimal threshold found = %.3f' % l_fbeta[best_thr_idx])
    print('Highest F-Bêta score = %.3f' % max(l_fbeta))
    
    return max(l_fbeta)


# In[16]:


def figure_fbeta_score (l_proba_thrs, l_fbeta, best_thr, model_label=None, return_fig = False):
    
    """ Draw the F-Bêta score figure for all probability thresholds tried. """
    
    if return_fig:
        # Create a matplotlib figure object.
        fig, ax = plt.subplots(1)
        
    # Plot the graph.
    if return_fig:
        ax.plot(l_proba_thrs, l_fbeta, label=model_label)
    else:
        plt.plot(l_proba_thrs, l_fbeta, label=model_label)
    
    # Plot a line at the best threshold found.
    plt.vlines(best_thr, ymin=0, ymax=max(l_fbeta), colors='k', linestyles='--')
     
    # Set other figures' parameters.
    plt.title('F-Bêta score = f(Probability thresholds)')
    plt.xlabel('Probability thresholds')
    plt.ylabel('F-Bêta score')
    plt.legend()
    
    if return_fig:
        return fig


# ## 8) Confusion matrix

# In[17]:


def figure_confusion_matrix (y_true, y_pred):
    
    """ Color the confusion matrix cases within a dataframe. """
    
    def cm_df_2x2_style (serie):
        
        """ Set the style of the confusion matrix. """
        
        cm_case = str(serie[0][0] + serie[0][1])
        if cm_case == 'TN':
            return ['background-color: #19C938', 'background-color: #FFC400'] # [TN, FP] | Sns deep green: #55A868, sns deep orange: #DD8453
        elif cm_case == 'FN':
            return ['background-color: #E8000B', 'background-color: #19C938'] # [FN, TP] | Sns deep red: #C44D52
    
    # Get the confusion matrix values.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Create the dataframe which will stores and show the 4 values.
    df = pd.DataFrame({'True negatives': ["TN: " + str(tn), "FP: " + str(fp)], 'True positives': ["FN: " + str(fn), "TP: " + str(tp)]},
                      index=['Predicted negatives', 'Predicted positives'])
    
    # Set the dataframe style.
    df = df.style.apply(cm_df_2x2_style)
    
    # Display the dataframe.
    display(df)


# ## 9) Classification indicators

# In[18]:


def calc_indicators (tp_fp_fn_tn):
    
    """ Calculate the classification indicators of: sensitivity, specificity, accuracy. """
    
    sensitivity = tp_fp_fn_tn[0] / (tp_fp_fn_tn[0] + tp_fp_fn_tn[2])
    specificity = tp_fp_fn_tn[3] / (tp_fp_fn_tn[3] + tp_fp_fn_tn[1])
    accuracy = (tp_fp_fn_tn[0] + tp_fp_fn_tn[3]) / np.sum(tp_fp_fn_tn)
    
    return [sensitivity, specificity, accuracy]


# ## 10) Job score

# In[19]:


def gain_norm (y_true, y_pred, fn_coeff = -10, fp_coeff = -1, tp_coeff = 0, tn_coeff = 0):

    """ Calculate and normalizes the job score which will be the main score used to measure models performances. """

    # Get the confusion matrix coeffs.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Gain.
    g = tp_coeff*tp + tn_coeff*tn + fp_coeff*fp + fn_coeff*fn
    
    # Maximum gain.
    g_max = tn_coeff*(fp + tn) + tp_coeff*(fn + tp)
    
    # Minimum gain.
    g_min = fp_coeff*(fp + tn) + fn_coeff*(fn + tp)
    
    # Normalized gain (MinMax).
    g_norm = (g - g_min) / (g_max - g_min)

    return g_norm


# In[20]:


def figure_job_score_curve (y_true, yhat_P, model_label, thr_line_vis = True, return_fig = False):
    
    """ Plot the job score curve function of a range of thresholds. """
    
    ### Configuration variables ###
    
    # Set the probability threshold range to try.
    l_proba_thrs = np.linspace(0, 1, num=101)
    
    # Get all set of predictions for each threshold.
    l_y_pred = get_y_pred_list(yhat_P)
    
    # Get the confusion matrix results for each set of predictions.
    np_tp, np_fp, np_fn, np_tn = np.array(get_tp_fp_fn_tn_lists(y_true, l_y_pred))
    
    # Calculate the optimal threshold.
    best_thr, best_thr_idx = opt_proba_thr(np_tp, np_fp, np_fn, np_tn, l_proba_thrs) 
    
    
    ### Caclculate the job score for each threshold within the set range ###
    
    l_g_norm = []
    for y_pred in l_y_pred:
        
        # Calculate the job score.
        g_norm = gain_norm(y_true, y_pred, fn_coeff=-10, fp_coeff=-1, tp_coeff=0, tn_coeff=0)
        
        # List of job scores.
        l_g_norm.append(g_norm)

    
    ### Plot and configure the figure ###
        
    if return_fig:
        
        # Create a matplotlib figure object.
        fig, ax = plt.subplots(1)
        
        # Plot the corresponding curve.
        ax.plot(l_proba_thrs, l_g_norm, label=model_label)
     
        # Plot a line at the best threshold found.
        ax.vlines(best_thr, ymin=0, ymax=max(l_g_norm), colors='k', linestyles='--', visible=thr_line_vis)
            
    else:
        
        # Plot the corresponding curve.
        plt.plot(l_proba_thrs, l_g_norm, label=model_label)
     
        # Plot a line at the best threshold found.
        plt.vlines(best_thr, ymin=0, ymax=max(l_g_norm), colors='k', linestyles='--', visible=thr_line_vis)
        
        
    # Set axis labels and the title.
    plt.xlabel("Classification threshold")
    plt.ylabel("Job score")
    plt.title("Job score")
    
    # Show the legend.
    plt.legend()
    
    if return_fig:
        return fig


# In[21]:


def scorer_fct (y_true, y_pred_proba_P, scorer = 'g_norm', verbose = True):
    
    """ Job scorer of scikit-learn. """
    
    # Set the probability threshold range to try.
    l_proba_thrs = np.linspace(0, 1, num=101)
    
    # Get all set of predictions for each threshold.
    l_y_pred = get_y_pred_list(y_pred_proba_P)
    
    # Get the confusion matrix results for each set of predictions.
    np_tp, np_fp, np_fn, np_tn = np.array(get_tp_fp_fn_tn_lists(y_true, l_y_pred))
    
    # Get the best threshold and its corresponding index with the list of the predictions sets.
    best_thr, idx = opt_proba_thr(np_tp, np_fp, np_fn, np_tn, l_proba_thrs, fn_cost_coeff = 10)
    best_thr = round(best_thr, 2)

    # Select the prediction set corresponding to the best threshold found.
    y_pred = l_y_pred[idx+1]

    
    # Select the scorer to use.
    if scorer == 'g_norm':
        # Calculate the normalized gain.
        score = gain_norm(y_true, y_pred, fn_coeff=-10, fp_coeff=-1, tp_coeff=0, tn_coeff=0)
        print(score)
    elif scorer == 'auroc':
        score = roc_auc_score(y_true, y_pred)
    elif scorer == 'fbeta':
        score = f_beta_score(y_true, y_pred, beta=3)
        
    
    if verbose:
        print('Best probability threshold:', best_thr)

    return score

# Make the function as a new scorer for sklearn.
g_norm_scorer = make_scorer(scorer_fct, scorer='g_norm', verbose=False)


# In[22]:


def scorer_w_thr_fct (y_true, y_pred_proba_P, scorer = 'g_norm', verbose = True):
    
    """ Function which make the scorer take into account the best probability threshold found for the current set of hyperparameters. """
    
    # Set the probability threshold range to try.
    l_proba_thrs = np.linspace(0, 1, num=101)
    
    # Get all set of predictions for each threshold.
    l_y_pred = get_y_pred_list(y_pred_proba_P)
    
    # Get the confusion matrix results for each set of predictions.
    np_tp, np_fp, np_fn, np_tn = np.array(get_tp_fp_fn_tn_lists(y_true, l_y_pred))
    
    # Get the best threshold and its corresponding index with the list of the predictions sets.
    best_thr, idx = opt_proba_thr(np_tp, np_fp, np_fn, np_tn, l_proba_thrs, fn_cost_coeff = 10)
    best_thr = round(best_thr, 2)

    # Select the prediction set corresponding to the best threshold found.
    y_pred = l_y_pred[idx+1]

    
    # Select the scorer to use.
    if scorer == 'g_norm':
        # Calculate the normalized gain.
        score = gain_norm(y_true, y_pred, fn_coeff=-10, fp_coeff=-1, tp_coeff=0, tn_coeff=0)
    elif scorer == 'auroc':
        score = roc_auc_score(y_true, y_pred)
    elif scorer == 'fbeta':
        score = f_beta_score(y_true, y_pred, beta=3)
        
    
    if verbose:
        print('Best probability threshold:', best_thr)

    return score, best_thr

# Make the function as a new scorer for sklearn.
g_norm_scorer_w_thr = make_scorer(scorer_w_thr_fct, scorer = 'g_norm', verbose=False)#, needs_proba=True)


# ## 11) MLFlow Tracker

# In[23]:


def mlflow_experiment_tracker (model_pl, df, model_label, X, y, l_tracked_vars = [],
                               eval_dataset = "_train_cv"):

    """ Store the model and all its relevant associated values and parameters. """
    
    ### [Display] MLFlow section delimiter start ###
    
    mlflow_section_title = " MLFlow tracker "
    print("\n" + "-" * int(50 - len(" MLFlow tracker ") / 2) + " MLFlow tracker " + "-" * int(50 - len(" MLFlow tracker ") / 2) + "\n")
    
    
    ### Required variables for some figures ###
    
    # List of the probability thresholds to try.
    l_proba_thrs = np.linspace(0, 1, num=201)

    # Get the predictions corresponding to each probability thresholds tried.
    l_y_pred = get_y_pred_list(l_tracked_vars[2], l_proba_thrs)

    # Get the corresponding TP, FP, FN and TN for each probability thresholds tried.
    np_tp, np_fp, np_fn, np_tn = np.array(get_tp_fp_fn_tn_lists(y, l_y_pred))
    
    # Calculate the F-bêta score for each probability thresholds tried.
    l_fbeta = []
    for y_pred in l_y_pred:
        fbeta = fbeta_score(y, y_pred, beta=2.5)
        l_fbeta.append(fbeta)
    
    
    ### MLFlow experiment and logs ###
    
    # Create or use an experiment folder.
    mlflow.set_experiment(model_label)
    
    # Save also the model scores and itself in a MLFlow log file.
    # NB: Log artifacts (ex: mlflow.log_artifacts(figure_dir, "figure_label")) could also be used to 
    #     add corresponding graphics to the logs of the model.
    with mlflow.start_run():
        
        # Log model.
        model_signature = infer_signature(X, y)
        try:
            mlflow.sklearn.log_model(model_pl, model_label + eval_dataset, signature=model_signature)
        except:
            try:
                mlflow.xgboost.log_model(model_pl, model_label + eval_dataset, signature=model_signature)
            except:
                try:
                    mlflow.lightgbm.log_model(model_pl, model_label + eval_dataset, signature=model_signature)
                except:
                    print("Troubles encountered to log the model.")
                    print()
        
        # Log model's hyperparameters.
        # NB: It seems that MLFlow has a character limit of 500 for the log_params method and 
        #     cannot handle the hyperameter list of the pipeline with XGBoost and MinMaxScaler which cumulate 736 charcaters.
        #        => If MLFlow cannot handle the size of the hyperparameters list, it is then recordeed as an artifact.
        try:
            mlflow.log_params(model_pl.get_params())
        except:
            try:
                mlflow.artifacts.load_dict(model_pl.get_params())
            except:
                # NB1: .artifact() cannot handle dictionaries so it is serialized in byte.
                # NB2: Whatever is the object type (byte, string...) it's too long to record...
                #mlflow.log_artifact(str(model_pl.get_params())) #pickle.dumps()
                print("Could not log the model hyperparams with MLFlow.")
                print()
                pass
                
        # Log metrics.
        mlflow.log_metric("best_thr", l_tracked_vars[3])
        mlflow.log_metric("job_score", l_tracked_vars[4])
        mlflow.log_metric("auroc", l_tracked_vars[5])
        mlflow.log_metric("fbeta", l_tracked_vars[6])
        mlflow.log_metric("process_time", l_tracked_vars[7])
        mlflow.log_metric("TP", l_tracked_vars[8][0])
        mlflow.log_metric("FP", l_tracked_vars[8][1])
        mlflow.log_metric("FN", l_tracked_vars[8][2])
        mlflow.log_metric("TN", l_tracked_vars[8][3])
        mlflow.log_metric("Sensitivity", l_tracked_vars[9][0])
        mlflow.log_metric("Specificity", l_tracked_vars[9][1])
        mlflow.log_metric("Accuracy", l_tracked_vars[9][2])
        
        # log figures.
        mlflow.log_figure(figure_roc(y, [l_tracked_vars[2]], [model_label], return_fig=True), "ROC+AUC.png") # AUROC.
        plt.close() # Prevent the figure to show in the notebook.
        #mlflow.log_figure(figure_sum_fp_coeff_fn(np_fp, np_fn, l_proba_thrs, l_tracked_vars[3], fn_cost_coeff=10, return_fig=True), "Inversed_job_score_curve.png") # Inversed gain.
        mlflow.log_figure(figure_sum_fp_coeff_fn(np_fp, np_fn, l_proba_thrs, l_tracked_vars[3], fn_cost_coeff=10, return_fig=True), "job_score_curve.png")
        plt.close()
        mlflow.log_figure(figure_density(y, l_tracked_vars[2], l_tracked_vars[3], return_fig=True), "Proba_density.png") # Probability density.
        plt.close()
        mlflow.log_figure(figure_fbeta_score(l_proba_thrs, l_fbeta, l_tracked_vars[3], return_fig=True), "Fbeta.png")
        plt.close()
        
        
    ### [Test] Save also the model alone with MLFlow ###
    # NB: Cannot overwrite an existing file.
    try:
        mlflow.sklearn.save_model(model_pl, MLFLOW_EXPORTS_MODEL_DIR_PATH + "_" + model_label)
    except:
        print("A folder with this name already exist.")
        print("Changing the storage folder name...")
        print()
        for i in range(100):
            change_folder_name = MLFLOW_EXPORTS_MODEL_DIR_PATH + "_" + model_label + "_" + str(i)
            try:
                mlflow.sklearn.save_model(model_pl, change_folder_name)
            except:
                pass
            
            # Show the saving status.
            if i > 100:
                print("It was not possible to save the model.")
                print()
            else:
                print("Model saved.")
                print()
                break
            
        pass
    
    
    ### [Display] MLFlow section delimiter end ###
    
    print("\n" + "-" * 100 + "\n")


# ## 12) Table to store all models' relevant values along notebooks

# In[24]:


def summarizing_table (df, l_vars, eval_dataset, l_col_labels):
    
    """ Update the table which sums up all models tried. """
    
    ### Variables unpacking ###
    # Labels.
    model_label_key, model_key, \
    yhat_train_key, yhat_test_key, \
    best_thr_train_key, best_thr_test_key, \
    g_norm_train_key, g_norm_test_key, \
    rocauc_train_key, rocauc_test_key, \
    fbeta_train_key, fbeta_test_key, \
    process_time_train_key, process_time_test_key, \
    cm_vals_train_key, cm_vals_test_key, \
    sensitivity_train_key, sensitivity_test_key, \
    specificity_train_key, specificity_test_key, \
    accuracy_train_key, accuracy_test_key = l_col_labels
    
    # Values.
    model_label, model, yhat, best_thr, g_norm, rocauc, fbeta, process_time, l_cm_vals, l_indicators = l_vars

    
    ### Select if the values corresponds to the validation set or the test set ###
    if eval_dataset == 'valid_set':
        dict_val = {yhat_train_key: yhat,
                    best_thr_train_key: best_thr,
                    g_norm_train_key: g_norm,
                    rocauc_train_key: rocauc,
                    fbeta_train_key: fbeta,
                    process_time_train_key: process_time,
                    cm_vals_train_key: l_cm_vals,
                    sensitivity_train_key: l_indicators[0],
                    specificity_train_key: l_indicators[1],
                    accuracy_train_key: l_indicators[2]                   
                   }

    else: # test_set
        dict_val = {yhat_test_key: yhat,
                    best_thr_test_key: best_thr,
                    g_norm_test_key: g_norm,
                    rocauc_test_key: rocauc,
                    fbeta_test_key: fbeta,
                    process_time_test_key: process_time,
                    cm_vals_test_key: l_cm_vals,
                    sensitivity_test_key: l_indicators[0],
                    specificity_test_key: l_indicators[1],
                    accuracy_test_key: l_indicators[2]
                   }
    
    dict_model = {model_label_key: model_label, model_key: model}
    

    ### Create or update the right line in the dataframe ###
    # NB: In python > 3.9 it is possible to merge dictionaries with "|" (the last dictionary takes the priority in conflicts).
    
    # Create a new row if it does not exist.
    if model_label not in df.index:
        print("Creating new entry...")
        df.loc[model_label] = dict_model | dict_val #[None] * df.shape[1]
        print("Done!")
        
    # Update the row.
    else:
        print("Updating entry...")
        df.loc[model_label] = df.loc[model_label].to_dict() | dict_model | dict_val 
        print("Done!")

    return df


# In[25]:


def update_sum_table (df, l_vars, get_csv_file, eval_dataset, main_scorer_val, l_col_labels = ['col0'],
                      main_scorer_train_label = 'Job_score_train', main_scorer_test_label = 'Job_score_test',
                      force_update = False):

    """ Create or update and save a new table to sum up all models tried. """
    
    # Update the csv file if the main score is higher.
    if get_csv_file:
        
        # Reload the csv file in a df (created or already loaded at the beginning of the notebook).
        df = pd.read_pickle(os.path.join(EXPORTS_MODELS_DIR_PATH, PKL_MODELS_FILE))#.set_index('Model_labels')
        
        # Remove the initializing row (first row of filled with None) if one of the added rows are already full.    
        if df.index[0] == '' and df.shape[0] > 1:
            for i in range(1, df.shape[0]):
                if sum(df.iloc[i].isna()) == 0:
                    df.drop(index=df.index[0], inplace=True) # Drop the now useless first row.
                    df = df.convert_dtypes() # Infere all dtypes to according to the data type under each column.
                    df = reduce_memory(df) # Tries to reduce the memory as much as possible by fine tuning columns' dtypes.
                    break
        
        # Check if the measures are from the test set or the train set.
        if eval_dataset == 'valid_set':
            main_scorer_label = main_scorer_train_label
        else:
            main_scorer_label = main_scorer_test_label
        
        # Check if the model_label entry is in the df.
        model_label = l_vars[0]        
        
        # Create a new entry.
        if model_label not in df.index:
            df = summarizing_table(df, l_vars, eval_dataset, l_col_labels)
            #df_to_csv_full(df)
            df.to_pickle(os.path.join(EXPORTS_MODELS_DIR_PATH, PKL_MODELS_FILE))
            print('The new informations have been saved in a new row.')
        
        # Check if the score is inferior from the one already stored in the csv file and update or not.
        elif df.loc[model_label, main_scorer_label] < main_scorer_val or pd.isnull(df.loc[model_label, main_scorer_label]) or force_update:
            df = summarizing_table(df, l_vars, eval_dataset, l_col_labels)
            #df_to_csv_full(df)
            df.to_pickle(os.path.join(EXPORTS_MODELS_DIR_PATH, PKL_MODELS_FILE))
            print('The row have been updated.')
        
        # Don't update if the new score is below the one in the csv file.
        else:
            print('The new score is inferior to the one already saved.')
            print('Dataframe not saved.')

            
    # Do not update the csv file (set by user).
    else: 
        df = summarizing_table(df, l_vars, eval_dataset, l_col_labels)

    return df


# ## 13) SHAP

# In[26]:


# NB: [OBSOLET] because of the shap.TreeExplainer()'s parameter model_output='probability' which convert raw shap values as 
#     logodd directly to thier odd counter part.
def logodd_to_odd (explanations, yhat, cat_class):
    
    """ Convert the logodd values to their odd counter parts. """
    
    # Initialize the explanation objects which will store the transformed values.
    explanation = copy.deepcopy(explanations)
    explanations_transformed = copy.deepcopy(explanations)
    
    # Store the length of each client's shape values in logodd.
    len_values = len(explanations[0])
    
    # Transform values.
    for i in range(len(explanations.values)):

        # Reformat the explanation attributes to their normal format.
        explanation.values = explanations.values[i].reshape(1, len_values)
                       
        # Select the probability to untransform (Select the value corresponding to the selected category's class).
        base_value = explanation.base_values[i]

        # Compute the original_explanation_distance to construct the distance_coefficient later on.
        original_explanation_distance = np.sum(explanation.values, axis=1)#[cat_class]

        # Get the untransformed base value (Odd).
        base_value_trf = 1 - expit(base_value) # = 1 - 1 / (1 + np.exp(-transformed_base_value))

        # Compute the distance between the model_prediction and the untransformed base_value.
        distance_to_explain = yhat[i][cat_class] - base_value_trf

        # The distance_coefficient is the ratio between both distances cat_class will be used later on.
        distance_coefficient = original_explanation_distance / distance_to_explain

        # Transform the original shapley values to the new scale (Odd scale).
        explanations_transformed.values[i] = explanation.values / distance_coefficient        
        
        # Finally reset the base_value as it does not need to be transformed.
        explanations_transformed.base_values[i] = base_value_trf       
        
    # Return the transformed array from the logodd to the odd scale.
    return explanations_transformed    


# In[27]:


def interpretability_shap (model, scaler, X_train, X_test, cat_class = 0): #customer_idx = None
      
    """ Get the SHAP values in their odd form with training of the explainer. """
        
    # Scale the train and the test set to fit the model pipeline inputs format.
    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    # Convert the numpy array X_test to a dataframe to associate the columns' labels to their corresponding values.
    # NB: This is required for some shap graphic which cannot get the columns' labels otherwise.
    X_test_norm = pd.DataFrame(X_test_norm, columns=X_test.columns)
    
    # Select data to interpret as global or local.
    #if customer_idx == None: # Global.
        #pd.DataFrame(X_test_norm, columns=X_test.columns)
    #else: # Local.
        #X_test_norm = pd.DataFrame(X_test_norm[customer_idx].reshape(1,-1), columns=X_test.columns)
    
    # Initialize time to measure the process duration.
    t0 = time()

    # Create the explainer model with TreeExplainer().
    # NB: model_output='probability' allows to display shap values on the probability scale (odd).
    explainer_shap = shap.TreeExplainer(model, X_train_norm, model_output='probability')

    # Get explanations (values = SHAP values, base_values = SHAP expected average values after its fit on X_train_norm, data = original data passed => X_test_norm).
    explanations = explainer_shap(X_test_norm)
    
    # If the negative class is chosen so the SHAP's values gotten for the positive class (1) needs to be adapated.
    if cat_class == 0:
        explanations.values = - explanations.values
        explanations.base_values = 1 - explanations.base_values

    # Measure the process time duration.
    delta_t = time() - t0
    
    return explanations, delta_t, explainer_shap


# In[28]:


def interpretability_shap_prod (X_test, scaler, explainer_shap, cat_class = 0):
    
    """ Get the SHAP values in their odd form without training the explainer. """
        
    # Scale the train and the test set to fit the model pipeline inputs format.
    #X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    # Convert the numpy array X_test to a dataframe to associate the columns' labels to their corresponding values.
    # NB: This is required for some shap graphic which cannot get the columns' labels otherwise.
    X_test_norm = pd.DataFrame(X_test_norm, columns=X_test.columns)
    
    # Select data to interpret as global or local.
    #if customer_idx == None: # Global.
        #pd.DataFrame(X_test_norm, columns=X_test.columns)
    #else: # Local.
        #X_test_norm = pd.DataFrame(X_test_norm[customer_idx].reshape(1,-1), columns=X_test.columns)
    
    # Initialize time to measure the process duration.
    #t0 = time()

    # Create the explainer model with TreeExplainer().
    # NB: model_output='probability' allows to display shap values on the probability scale (odd).
    #explainer_shap = shap.TreeExplainer(model, X_train_norm, model_output='probability')

    # Get explanations (values = SHAP values, base_values = SHAP expected average values after its fit on X_train_norm, data = original data passed => X_test_norm).
    explanations = explainer_shap(X_test_norm)
    
    # If the negative class is chosen so the SHAP's values gotten for the positive class (1) needs to be adapated.
    if cat_class == 0:
        explanations.values = - explanations.values
        explanations.base_values = 1 - explanations.base_values

    # Measure the process time duration.
    #delta_t = time() - t0

    return explanations


# ## 14) Any python object serialization to string and deserialization
# 
# Very useful to transfer any python object in json format across APIs.

# In[29]:


def obj_to_txt (obj):
    
    """ Serialize an object into a plain text. """
    
    message_bytes = pickle.dumps(obj)
    base64_bytes = base64.b64encode(message_bytes)
    txt = base64_bytes.decode('ascii')
    
    return txt


# In[30]:


def txt_to_obj (txt):
    
    """ De-serialize an object from its plain text serialization counter part. """
    
    base64_bytes = txt.encode('ascii')
    message_bytes = base64.b64decode(base64_bytes)
    obj = pickle.loads(message_bytes)
    
    return obj

