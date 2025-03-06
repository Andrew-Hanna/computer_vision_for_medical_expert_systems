import streamlit as st
from tensorflow.keras.models import load_model
from logic import convert_model_output, forward_chaining_with_explanation, rules_with_explanation, preprocess_image, ask_symptoms
import numpy as np

# Filepath to the saved model
model_path = 'covid_detection_model.h5'

# Load the model
model = load_model(model_path)

# Check the model summary to confirm it was loaded correctly
model.summary()

# Title of the app
st.title("Welcome to Covid Detection Expert System")
st.write("Provide input for the system to analyze.")

# Select mode
option = st.selectbox(
    "Select which mode you want to use:",
    ("Only Image", "Only Text", "Image and Text"),
)

st.write("You selected:", option)


if option == "Only Image":
    st.write("You selected Image Mode.")
    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        print(uploaded_file.name)
        target_size = (100, 100)
        image = preprocess_image(uploaded_file.name, target_size)
        predic = model.predict(image)
        prediction = np.argmax(predic, axis=1)
        print(prediction)
        symptoms = convert_model_output(int(prediction),0)
        print(symptoms)
        diagnosis, explanation =  forward_chaining_with_explanation(symptoms)
        st.write(f"Diagnosis: {diagnosis}") 
        st.write(f"Explanation: {explanation}")

# elif option == "Only Text":
#     st.write("You selected Text Mode.")
#     user_text = st.text_area("Enter your text:")
#     if user_text:
#         st.write(f"Provided text: {user_text}")
#         print(type(user_text))
#         symptoms = ask_symptoms(user_text)
#         print(symptoms)

#         diagnosis, explanation =  forward_chaining_with_explanation(symptoms)
        

#         print(f"Diagnosis: {diagnosis}")
#         print(f"Diagnosis: {explanation}")
#         symptoms = {}
elif option == "Only Text":
    st.write("You selected Text Mode.")
    user_text = st.text_input("Enter your text:")
    
    # Add a button to start the system
    if st.button("Start System"):
        if user_text:  # Ensure there's text to process
            st.write(f"Provided text: {user_text}")
            print(type(user_text))
            
            symptoms = ask_symptoms(user_text)
            print(symptoms)
            
            diagnosis, explanation = forward_chaining_with_explanation(symptoms)
            st.write(f"Diagnosis: {diagnosis}") 
            st.write(f"Explanation: {explanation}")
            
            print(f"Diagnosis: {diagnosis}")
            print(f"Explanation: {explanation}")
            
            # Reset symptoms
            symptoms = {}
        else:
            st.warning("Please enter text before starting the system.")


elif option == "Image and Text":
    st.write("You selected Image and Text Mode.")
    user_text = st.text_input("Enter your text:")
    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        print(uploaded_file.name)
        target_size = (100, 100)
        image = preprocess_image(uploaded_file.name, target_size)
        predic = model.predict(image)
        prediction = np.argmax(predic, axis=1)
        print(prediction)
        symptoms = convert_model_output(int(prediction),0)
        print(symptoms)
        diagnosis, explanation =  forward_chaining_with_explanation(symptoms)
        st.write(f"Diagnosis: {diagnosis}") 
        st.write(f"Explanation: {explanation}")
        if user_text:
            st.write(f"Provided text: {user_text}")
            
            symptoms = ask_symptoms(user_text)
            print(symptoms)


    

    # if uploaded_file is not None:
    #     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    # if user_text:
    #     st.write(f"Provided text: {user_text}")
        
    #     symptoms = ask_symptoms(user_text)
    #     print(symptoms)

    #     diagnosis, explanation =  forward_chaining_with_explanation(symptoms)
        

        print(f"Diagnosis: {diagnosis}")
        print(f"Diagnosis: {explanation}")

