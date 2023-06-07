            ### Importation of libraries ###

# Allow to stop the script at any moment with sys.exit().
import sys

# Import streamlit.
import streamlit as st

# Libraries for APIs communication.
import requests
import json

# Files' path.
import os.path

# Save and load files.
import csv
import pickle

# Data manipulations.
import numpy as np
import pandas as pd

# SHAP.
import shap
#import streamlit.components.v1 as components
from streamlit_shap import st_shap # NB: SHAP wrapper in order to replace shap.initjs() which was not working because of javascript library not loading.
#shap.initjs()
#shap.getjs()

# Visualizations.
import matplotlib.pyplot as plt
#import seaborn as sns
import plotly.graph_objects as go

# Custom functions.
from shared_functions import txt_to_obj # Serialize SHAP for API transfer.



            ### Global files paths and names ###
            
# NB!: On windows local host \ or / as path separator does not matter as long as the prefix "r" is present at the beginning.
#      However, on linux machines (most probably the environment which will be used for deployment),
#      it has to be / whatever if there is a repertory prefix or not.          
IMPORT_DATA_DIR_PATH = r'Exports/Preprocessed_data'
IMPORT_MODELS_SUM_DIR_PATH = r'Exports/Models/Selected'
API_SERVER_PATH = json.load(open('urls.json', 'r'))['on_line']['backend_url'] #OnLine: 'https://hidden-savannah-70356.herokuapp.com/' # OnLan: 'http://127.0.0.1:5000/'

PKL_MODELS_SUM_FILE = 'selected_model_infos.pkl'




            ### Functions ###

@st.cache_data() # Store customers data in cache for faster access.
def load_data ():

    """ Load customers data."""

    #df = pd.read_csv(os.path.join(IMPORT_DATA_DIR_PATH, 'preprocessed_data_new_customers.csv'))
    df = pd.read_pickle(os.path.join(IMPORT_DATA_DIR_PATH, 'preprocessed_data_new_customers.pkl'))
        
    return df

def get_prediction (customer_id):
    
    """ Get the model predictions over the observed customer (probability of non payment default) from the predictive API."""
    
    # Set the url and the parameters to get the predictions. 
    url_api_predictions = API_SERVER_PATH + '/api/predictions/%i' % int(customer_id)
    #var_dict = {'customer_id': selected_customer_id}

    # Make the request and get the corresponding response in a json format.
    response_pred = requests.get(url_api_predictions)
    
    # Extract the result of the prediction and convert it back from list to a np.array format.
    yhat_customer = np.array(response_pred.json())
  
    return yhat_customer
   
def get_shap_explanations (customer_id, cat_class = 0):

    """ Get the SHAP explanations over the observed customer (SHAP values indicated as probability of non payment default)
        from the SHAP API. """

    # Set the url and the required parameters to get the SHAP explanations. 
    url_api_shap_expl = API_SERVER_PATH + '/api/interpretations/%i/%i' % (int(customer_id), int(cat_class))
    
    # Make the request and get the corresponding response in a json format.
    response_expl = requests.get(url_api_shap_expl)
       
    # Extract the results from the response.
    shap_expl = txt_to_obj(response_expl.text)
       
    return shap_expl


#def st_shap(shap_plot, height=10):
#    shap_html = f"<head>{shap.getjs()}</head><body>{shap_plot.html()}</body>"
#    components.html(shap_html, height=height)




            ### Global variables ###

# Class to consider (0: Negative customer => reliable. 1 for the opposite).
# NB: This allows to focus on negative or positive customers with the appropriated default probability (negative, positive).
#     Put the focus on negative probabilities by default appears to me closer to the common sense of people mind.
#       => Results are easier to understand and explain.
CAT_CLASS = 0




            ### Data loading ###

# Load customers' data and store it in cache for quicker use.
df = load_data()

# Extract the customer IDs.
l_customer_id = df['SK_ID_CURR'].to_list()

# Add None as default value (In order to respect RGPD).
l_customer_id.insert(0, 'Average')




            ### Dashboard building ###

        ## Section 0: Dashboard title and description ##

# Set the dashboard title.
st.markdown("<h1 style='text-align: center; color: black'> Management recommendations of customer's applications </h1>",
            unsafe_allow_html=True)

# Dashboard description.
st.markdown("<h5 style='text-align: justify; color: black'> This dashboard allows to get the recommendation on the way to manage\
            the submitted customer's application and easily understand the main reasons which led to this result.</h5>",
            unsafe_allow_html=True)



        ## Section 1: Customer selection and final prediction ##

# Section title.
st.markdown("<h2 style='text-align: center; color: black'> Recommendation & customer position </h2>", unsafe_allow_html=True)

# Section description.
st.markdown("<h6 style='text-align: justify; color: black'> This section shows the recommendation and the customer's application\
            position among the average and the required standards.</h6>", unsafe_allow_html=True)

    # Create the a selection button among customer IDs #

# Grid level 1: Set the grid in which place the selection button.
col1, col2, _ = st.columns([1,3,1])

# Place and create the a selection button between customers.
# NB: For RGPD: index = None => None as default customer ID.
with col1:
    selected_customer_id = st.selectbox("Select customer ID", l_customer_id) # For a default value: index=<default_value_idx>
    
    # For convinience if the user select None the empty variable is filled with -1 instead.
    if selected_customer_id == 'Average':
        selected_customer_id = 0
    

    # Get customer information #

# Get the SHAP explanations of the prediction for the selected customer.
shap_expl = get_shap_explanations(selected_customer_id, cat_class=CAT_CLASS)

# Get the prediction recommendation for the selected customer.
if selected_customer_id != 0:
    yhat_customer = get_prediction(selected_customer_id)
else:
    yhat_customer = [shap_expl.base_values[0], 1 - shap_expl.base_values[0]]

# Get the probability threshold that the model uses to accept or deny an application.
df_models_sum = pd.read_pickle(os.path.join(IMPORT_MODELS_SUM_DIR_PATH, PKL_MODELS_SUM_FILE))
proba_thr = df_models_sum.iloc[-1, 3]


    # Sentence to show the result of the model prediction #

# Within the same grid as previously, add a colored sentence which clearly show if the model accepted or denied the customer's application.
with col2:
    if yhat_customer[CAT_CLASS] > proba_thr:
        st.markdown("<h2 style='text-align: center; color: lightgreen'> Application accepted </h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='text-align: center; color: salmon'> Application denied </h2>", unsafe_allow_html=True)
    


        ## Section 2: Gauge which quickly sum up the status of the customer's application ##

# Grid level 2: Set a new grid which will follow the previous one.
col1, col2, col3 = st.columns([1,3,1])

# Set the gauge figure.
with col2:
    fig = go.Figure(go.Indicator(mode = "gauge+number+delta",
                                 value = yhat_customer[CAT_CLASS],
                                 domain = {'x': [0, 1], 'y': [0, 1]},
                                 title = {'text': "Customer's repay probability"},
                                 delta = {'reference': proba_thr, 'increasing': {'color': "lightgreen"}, 'decreasing': {'color': "salmon"}},
                                 gauge = {'axis': {'range': [0, 1]},
                                          'bar': {'color': "royalblue"},
                                          'borderwidth': 2,
                                          'bordercolor': "black",
                                          'steps': [{'range': [0, proba_thr], 'color': "salmon"}, {'range': [proba_thr, 1], 'color': "lightgreen"}],
                                          'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 1, 'value': shap_expl.base_values[0]}},
                                ))
    fig.update_layout(width=400, height=400)
    st.plotly_chart(fig, use_container_width=False)

# Set the Help button which explains the gauge.
with col3:
#    m = st.markdown("""<style>div.stButton > button:first-child {background-color: salmon;}</style>""", unsafe_allow_html=True)
#    b = st.button("Trustless range")

    # Help button for the gauge figure.
    description = "The gauge quickly highlight the predicted probability that the customers repays its credit in blue.\
                   Red: Trustless range.\
                   Green: Trustful range.\
                   White bar: Average trust among all customers.\
                   Colored triangle: How far the customer's repay probability is from the probability threshold.\
                   NB: When global is selected the blue bar represents the average of all customers' applications."
    st.button("?", help=description, use_container_width=False)
    


        ## Section 3: Explanations of the prediction choice ##

# Section title.
st.markdown("<h2 style='text-align: center; color: black'> Influence of the features </h2>", unsafe_allow_html=True)

# Section description.
st.markdown("<h6 style='text-align: justify; color: black'> This section highlights the features with the highest influence to\
            easily explain the reason which led to the resulting recommendation. </h6>", unsafe_allow_html=True)


    # Summarized #
    
if selected_customer_id != 0:

    # Grid level 3: Set the 3rd grid level.
    _, col2, col3 = st.columns([1, 1, 3.75])

    # Set the title of the figure with its description.
    with col3:
        description = "This figure quickly highlights the main features which influenced the most the model prediction.\
                       NB: The 2nd figure down below show the same information at a more detailed format."
        st.markdown("<h3 style='text-align: center; color: black;'> Summarized </h3>", unsafe_allow_html=True, help=description)
        

    # Draw the interactive SHAP line force plot.
    st_shap(shap.plots.force(shap_expl[0]))


    # Detailed #

st.markdown("<h3 style='text-align: center; color: black;'> Detailed </h3>", unsafe_allow_html=True)

# Select the number of top features (the most important ones for the model according to SHAP) to show.
# NB: st_shap has not been used in this section because it was impossible to manage
#     the size and shape winthin the dashboard of those figures drawn this way.
description = "Select the number of features to show in both figures down below."
top_ft = st.slider("Number of features to show", min_value=1, max_value=len(shap_expl[0].values), value=10, step=1,
                   help=description, disabled=False, label_visibility="visible")

# Grid level 4: Set the 4th grid level.
col1, col2 = st.columns(2, gap="small")

# Draw the absolute SHAP values as probabilities.
fig, ax = plt.subplots() # Required to set the shap.plots as a matplotlib figure (It won't display properly otherwise with st.pyplot).
with col1:

    # Title and description of the figure.
    description = "This graphic shows the importance of each feature toward the model prediction in term of absolute values."
    st.markdown("<h4 style='text-align: right; color: black;'> Features importance </h4>", unsafe_allow_html=True, help=description)
    
    # Turn the SHAP graphical object to a displayable matplotlib object.
    # NB: This step is required because this SHAP figure is considered as NoneType and would generate warnings or errors as is.
    #       => Convert it as a matplolib object, solve the problem.
    ax = shap.plots.bar(shap_expl, max_display=top_ft+1, show=False)
    
    # Draw the figure.
    st.pyplot(fig)

# Draw the detailed force plot of SHAP values as probabilities.
fig, ax = plt.subplots() # Reinitialize the figure environment in order to avoid any superimpose with the previous graphic.
with col2:

    if selected_customer_id != 0:
    
        # Title and description of the figure.
        description = "This graphic shows in which measure each feature contributed to lead to the model prediction result\
                       from the average customers' probability to see their application accepted \
                       (NB: The red line shows the limit above which the model will recommend the acceptation of the customer's application)."
        st.markdown("<h4 style='text-align: right; color: black;'> Features influence </h4>", unsafe_allow_html=True, help=description)

        # Add the probability threshold on the figure which will decide if the customer's application should be accepted or denied.
        ax.axvline(x=proba_thr, color='r', linestyle='--') # Add before ax =.

        # Turn the SHAP graphical object to a displayable matplotlib object.
        # NB: This step is required because this SHAP figure is considered as NoneType and would generate warnings or errors as is.
        #       => Convert it as a matplolib object, solve the problem.
        ax = shap.plots.waterfall(shap_expl[0], max_display=top_ft+1, show=False)
       
    else:
        # Title and description of the figure.
        description = "This figure shows in which measure the values of a feature influence the feature in positive or negative way\
                       to accept or deny an application by the model. In addition, it also give an idea about the customer's distribution\
                       within each feature."
        st.markdown("<h4 style='text-align: right; color: black;'> Values influence </h4>", unsafe_allow_html=True, help=description)
           
        # Turn the SHAP graphical object to a displayable matplotlib object.
        # NB: This step is required because this SHAP figure is considered as NoneType and would generate warnings or errors as is.
        #       => Convert it as a matplolib object, solve the problem.
        ax = shap.plots.beeswarm(shap_expl, max_display=top_ft+1, show=False)
       
    # Draw the final figure
    st.pyplot(fig)



        ## Section 4: Display all data concerning the customer that the model take as input ##

# Stop the script in case no customer ID was chosen.
if selected_customer_id == 0:
    sys.exit()
    
# Section title.
st.markdown("<h2 style='text-align: center; color: black'> Customer's data </h2>", unsafe_allow_html=True)

# Section description.
st.markdown("<h6 style='text-align: justify; color: black'> This section shows all customer's data. </h6>", unsafe_allow_html=True)

# Grid level 5: Set the 5th grid level.
_, col2, _ = st.columns([1,3,1])
with col2:

    # Title and description of the figure.
    description = "This table shows all customer's data used for the model prediction."
    st.markdown("<h4 style='text-align: center; color: black;'> Customer's data </h4>", unsafe_allow_html=True, help=description)

    # Display the raw data corresponding to the selected customer.
    st.dataframe(df[df['SK_ID_CURR'] == selected_customer_id].set_index('SK_ID_CURR').T, use_container_width=True)