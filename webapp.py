import numpy as np
import pickle
import streamlit as st 

# Load the model
model_path = '/Users/shmanik/Downloads/web link/predictor.sav'  
# Update the path to your model file
loaded_model = pickle.load(open(model_path, 'rb'))

# Creating a function for Prediction
def heartdisease_prediction(input_data):
    # Changing the input_data to numpy array and converting to float
    input_data_as_float = [float(x) for x in input_data]
    input_data_as_numpy_array = np.asarray(input_data_as_float)
    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    
    if prediction[0] == 'Yes':
        return 'Has heart disease'
    else:
        return 'Does not have heart disease'

def main():
    # Giving a title
    st.title('Patient Heart Disease Predictor Application')
    
    # Getting the input data from the user
    physical_health_days = st.text_input('For how many days during the past 30 days was your physical health not good?', '0.0')
    mental_health_days = st.text_input('For how many days during the past 30 days was your mental health not good?', '0.0')
    sleep_hours = st.text_input('Sleep Hours-Average hours of sleep per day', '0.0')
    bmi = st.text_input('BMI(Normal weight (18.5 <= BMI < 25.0))', '0.0')
    #general_health_excellent = st.radio('General Health - Excellent', [False, True])
    #general_health_good = st.radio('General Health - Good', [False, True])
    #general_health_poor = st.radio('General Health - Poor', [False, True])
    SmokerStatus_current_smoker = st.radio('Smoker Status - Current Smoker', [False, True])
    SmokerStatus_former_smoker = st.radio('Smoker Status - Former Smoker', [False, True])
    SmokerStatus_never_smoke = st.radio('Smoker Status - Never Smoke', [False, True])
    AlcoholDrinkers_No = st.radio('AlcoholDrinkers - No', [False, True])
    AlcoholDrinkers_yes = st.radio('AlcoholDrinkers - Yes', [False, True])
    physical_activities_no = st.radio('Physical Activities - No', [False, True])
    physical_activities_yes = st.radio('Physical Activities - Yes', [False, True])
    had_angina_no = st.radio('Had Angina(History of angina) - No', [False, True])
    had_angina_yes = st.radio('Had Angina(History of angina) - Yes', [False, True])
    had_stroke_no = st.radio('Had Stroke(History of stroke) - No', [False, True])
    had_stroke_yes = st.radio('Had Stroke(History of stroke) Yes', [False, True])
    had_asthma_no = st.radio('Had Asthma(History of Asthma) - No', [False, True])
    had_asthma_yes = st.radio('Had Asthma(History of Asthma) - Yes', [False, True])
    had_copd_no = st.radio('Had COPD(Chronic obstructive pulmonary disease) - No', [False, True])
    had_copd_yes = st.radio('Had COPD(Chronic obstructive pulmonary disease) - Yes', [False, True])
    #general_health = st.selectbox('How would you like to be contacted?',('Good', 'Excellent', 'poor'))
    had_depressive_disorder = st.radio('Had Depressive Disorder(History of depressive disorder) - No', [False, True])
    
    # Code for Prediction
    if st.button('Predict if patient has heart disease'):
        prediction = heartdisease_prediction([physical_health_days, mental_health_days,
                                              sleep_hours, bmi, 
                                              SmokerStatus_current_smoker,SmokerStatus_former_smoker,
                                              SmokerStatus_never_smoke,
                                              AlcoholDrinkers_No, AlcoholDrinkers_yes, physical_activities_no,
                                              physical_activities_yes, had_angina_no,
                                              had_angina_yes, had_stroke_no, had_stroke_yes,
                                              had_asthma_no, had_asthma_yes, had_copd_no,
                                              had_copd_yes,had_depressive_disorder])
        st.success(f'Predicted patient heart disease status is : {prediction}')

if __name__ == '__main__':
    main()


#This code is for building a simple web application using Streamlit for predicting whether a patient has heart disease based on input features. Here's a breakdown of the code:

#Imports:

#numpy (as np): For numerical computations.
#pickle: For loading the pre-trained machine learning model from a file.
#streamlit as st: For creating the web application interface.
#Load Model:

#The path to the pre-trained model file (predictor.sav) is specified.
#The model is loaded into memory using pickle.load().
#Prediction Function:

#heartdisease_prediction(): This function takes input data as a list of features, converts it into a numpy array, reshapes it, and then passes it through the loaded model to make predictions. The prediction result ('Yes' or 'No') is returned.
#Main Function:

#main(): This function is the entry point of the Streamlit application.
#It creates an interface for users to input data.
#The input fields include physical health days, mental health days, sleep hours, BMI, and several other binary options related to general health conditions and medical history.
#When the user clicks the "Predict if patient has heart disease" button, the input data is passed to the heartdisease_prediction() function, and the predicted result is displayed as a success message.
#Streamlit App Execution:

#The main() function is called when the script is executed (if _name_ == '_main_': main())