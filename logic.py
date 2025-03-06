import numpy as np
import cv2

def convert_model_output(model_output, num ):
    """
    Convert model numeric output to descriptive labels with an 'extra' field.
    
    :param model_output: Numeric output from the model (0, 1, 2, 3).
    :return: Dictionary with descriptive label and an 'extra' field set to True.
    """
    case_codes = {0: 'Normal', 1: 'COVID', 2: 'Lung_Opacity', 3: 'Viral Pneumonia'}
    
    # Retrieve the corresponding label
    label = case_codes.get(model_output, 'Unknown')
    
    # Return the label with an 'extra' field set to True
    if num == 0: 
        return { 'chest_xray'  : label,
            "only": True,}
    return { 'chest_xray'  : label,
            }

    # Rule-based expert system
rules_with_explanation = [
    {'conditions': {'fever': True, "cough": True}, 'conclusion': 'flu', 'explanation': {"fever and cough are common flu symptoms."}},
    {'conditions': {'nausea': True, "vomiting": True}, 'conclusion': 'food poisoning', 'explanation': {"Nausea and vomiting are typical for food poisoning."}},
    {'conditions': {'fever': True, "shortness_of_breath": True, "fatigue": True}, 'conclusion': 'COVID', 'explanation': {"These symptoms are indicative of COVID-19."}},
    {'conditions': {'chest_xray': 'Normal', 'fever': True, "cough": True, "shortness_of_breath": True}, 'conclusion': 'Lung_Opacity', 'explanation': {"Chest X-ray shows lung opacity, indicating possible pneumonia."}},
    {'conditions': {'chest_xray': 'Normal', 'fever': True, "cough": True, "fatigue": True}, 'conclusion': 'Viral Pneumonia', 'explanation': {"Chest X-ray shows signs of viral pneumonia."}},
    {'conditions': {'fever': True, "headache": True, "muscle_aches": True}, 'conclusion': 'viral infection', 'explanation': {"Fever, headache, and muscle aches are indicative of a viral infection."}},
    {'conditions': {'fever': True, "abdominal_pain": True, "diarrhea": True}, 'conclusion': 'gastroenteritis', 'explanation': {"Abdominal pain, fever, and diarrhea suggest gastroenteritis."}},
    {'conditions': {'fever': True, "jaundice": True}, 'conclusion': 'hepatitis', 'explanation': {"Fever and jaundice indicate possible hepatitis."}},
    {'conditions': {'fever': True, "sore_throat": True, "swollen_lymph_nodes": True}, 'conclusion': 'streptococcal throat infection', 'explanation': {"These symptoms are indicative of a streptococcal throat infection."}},
    {'conditions': {'fever': True, "chills": True, "body_aches": True}, 'conclusion': 'influenza', 'explanation': {"Fever, chills, and body aches are common symptoms of influenza."}},
    {'conditions': {'fever': True, "rash": True, "joint_pain": True}, 'conclusion': 'chikungunya', 'explanation': {"Fever, rash, and joint pain suggest chikungunya."}},
    {'conditions': {'fever': True, "confusion": True, "altered_mental_status": True}, 'conclusion': 'meningitis', 'explanation': {"These symptoms are indicative of meningitis."}},
    {'conditions': {'fever': True, "back_pain": True, "urinary_urgency": True}, 'conclusion': 'urinary tract infection (UTI)', 'explanation': {"Fever, back pain, and urinary urgency suggest a urinary tract infection."}},
    {'conditions': {'fever': True, "exposure_to_infected_person": True}, 'conclusion': 'exposure to contagious disease', 'explanation': {"Fever and known exposure to an infected person suggest a contagious disease."}},
    {'conditions': {'fever': True, "recent_travel": True, "cough": True}, 'conclusion': 'travel-related illness', 'explanation': {"Fever, cough, and recent travel indicate a travel-related illness."}},
    {'conditions': {'fever': True, "chest_pain": True, "cough": True}, 'conclusion': 'pulmonary embolism', 'explanation': {"These symptoms may indicate a pulmonary embolism."}},
    {'conditions': {'fever': True, "pain_urination": True}, 'conclusion': 'urethritis', 'explanation': {"Fever and painful urination suggest urethritis."}},
    {'conditions': {'fever': True, "night_sweats": True, "cough": True}, 'conclusion': 'tuberculosis (TB)', 'explanation': {"Fever, night sweats, and cough are indicative of tuberculosis."}},
    {'conditions': {'fever': True, "exposure_to_animals": True, "rash": True}, 'conclusion': 'zoonotic disease', 'explanation': {"Fever and exposure to animals suggest a zoonotic disease."}},
    {'conditions': {'fever': True, "sudden_shortness_of_breath": True, "chest_pain": True}, 'conclusion': 'heart attack', 'explanation': {"These symptoms may indicate a heart attack."}},
    {'conditions': {'chest_xray': 'Normal',"only": True,}, 'conclusion': 'normal', 'explanation': {"Chest X-ray appears normal, no visible abnormalities."}},
    {'conditions': {'chest_xray': 'Lung_Opacity',"only": True,}, 'conclusion': 'pneumonia', 'explanation': {"Chest X-ray shows lung opacity, indicating possible pneumonia."}},
    {'conditions': {'chest_xray': 'Viral Pneumonia',"only": True,}, 'conclusion': 'viral pneumonia', 'explanation': {"Chest X-ray shows signs of viral pneumonia."}},
    {'conditions': {'chest_xray': 'COVID',"only": True,}, 'conclusion': 'COVID', 'explanation': {"Chest X-ray shows features consistent with COVID-19."}},
    { 'conditions': {'chest_xray': 'Normal', 'fever': False, "cough": False, "shortness_of_breath": False}, 'conclusion': 'Normal', 'explanation': {"Chest X-ray is clear. No signs of abnormality detected."} },
    { 'conditions': {'chest_xray': 'Viral Pneumonia', 'fever': True, "cough": True, "fatigue": True}, 'conclusion': 'Viral Pneumonia', 'explanation': {"Chest X-ray shows signs of viral pneumonia."} },
    { 'conditions': {'chest_xray': 'COVID', 'fever': True, "cough": True, "shortness_of_breath": True, "loss_of_taste_or_smell": True}, 'conclusion': 'COVID-19', 'explanation': {"Chest X-ray shows ground-glass opacities, which are indicative of COVID-19 infection."} }, 
    { 'conditions': {'chest_xray': 'Lung_Opacity', 'fever': True, "cough": True, "shortness_of_breath": True}, 'conclusion': 'Lung Opacity', 'explanation': {"Chest X-ray shows lung opacity, indicating possible pneumonia."} }
]

def forward_chaining_with_explanation(symptoms):
    for rule in rules_with_explanation:
        if all(symptoms.get(cond, False) == value for cond, value in rule['conditions'].items() ):
            return rule["conclusion"], rule["explanation"]
    return "No diagnosis found .", "No sufficient evidence."


def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image = cv2.resize(image, target_size) 
    image = image.astype('float32') / 255.0 # rescale 
    image = np.expand_dims(image, axis=0) # add batch dimension return image
    return image 

def ask_symptoms(symptoms):
    print(type(symptoms))
    lst = symptoms.split(',')
    newsymptoms = {}
    
    for symptom in lst:
        # symptom = input(f"Enter symptom {i+1} (comma separated): ").strip().lower()
        newsymptoms[symptom] = True  # Add symptom to the dictionary
    return newsymptoms