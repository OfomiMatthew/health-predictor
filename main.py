from flask import Flask,request,render_template,redirect,flash,url_for# type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import pickle

app = Flask(__name__)
app.secret_key = 'your-very-secure-secret-key'

# ------ loading dataset

symptoms_df = pd.read_csv('dataset/symtoms_df.csv')
precautions_df = pd.read_csv('dataset/precautions_df.csv')
workout_df = pd.read_csv('dataset/workout_df.csv')
description_df = pd.read_csv('dataset/description.csv')
medication_df = pd.read_csv('dataset/medications.csv')
diet_df = pd.read_csv('dataset/diets.csv')

svc = pickle.load(open('models/svc.pkl','rb'))





# def helper(disease):
#     describe = description_df[description_df['Disease'] == disease]['Description']
#     describe = " ".join([word for word in describe])

#     precaution = precautions_df[precautions_df['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
#     precaution = [col for col in precaution.values]

#     medication = medication_df[medication_df['Disease'] == disease]['Medication']
#     medication = [med for med in medication.values]

#     diets = diet_df[diet_df['Disease'] == disease]['Diet']
#     diets = [diet for diet in diets.values]

#     workout = workout_df[workout_df['disease'] == disease] ['workout']


#     return describe,precaution,medication,diets,workout




# symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
# diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}
def helper(disease):
    disease = disease.strip().lower()  # Normalize input
    describe = description_df[description_df['Disease'].str.lower() == disease]['Description']
    if describe.empty:
        raise KeyError(f"Disease '{disease}' not found in the description dataset.")
    describe = " ".join(describe)

    precaution = precautions_df[precautions_df['Disease'].str.lower() == disease]
    if precaution.empty:
        raise KeyError(f"Disease '{disease}' not found in the precautions dataset.")
    precaution = [precaution[col].values[0] for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]

    medication = medication_df[medication_df['Disease'].str.lower() == disease]['Medication']
    medication = [med for med in medication.values]

    diets = diet_df[diet_df['Disease'].str.lower() == disease]['Diet']
    diets = [diet for diet in diets.values]

    workout = workout_df[workout_df['disease'].str.lower() == disease]['workout']
    if workout.empty:
        workout = ["No workout recommendation available."]
    return describe, precaution, medication, diets, workout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}


# def get_predicted_value(patient_symptoms):
#     input_vector = np.zeros(len(symptoms_dict))
#     print(f'input vector: {input_vector}')

#     for item in patient_symptoms:
#         input_vector[symptoms_dict[item]] = 1
#     return diseases_list[svc.predict([input_vector])[0]]

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item not in symptoms_dict:
            raise KeyError(f"Symptom '{item}' not found in the dictionary.")
        input_vector[symptoms_dict[item]] = 1
    prediction_index = svc.predict([input_vector])[0]
    if prediction_index not in diseases_list:
        raise KeyError(f"Predicted disease index '{prediction_index}' not found in diseases list.")
    return diseases_list[prediction_index]

    




@app.route('/')
def home():
    return render_template('index.html')



# @app.route('/predict',methods=['GET','POST'])
# def predict():
#     if request.method == 'POST':
#         symptoms = request.form.get('symptoms')
#         user_symptoms = [s.strip() for s in symptoms.split(',')]
#         user_symptoms = [sym.strip("[]' ") for sym in user_symptoms]
        
#         # Get the predicted disease and related data
#         try:
#             predicted_disease = get_predicted_value(user_symptoms)
#             describe, precaution, medication, diets, workout = helper(predicted_disease)
            
#             # Enumerate the lists for precaution, medication, diets, and workout
#             precaution_enumerated = [(index + 1, prec) for index, prec in enumerate(precaution[0])]
#             medication_enumerated = [(index + 1, med) for index, med in enumerate(medication[0][1:-1].replace("'", "").split(", "))]
#             diet_enumerated = [(index + 1, diet) for index, diet in enumerate(diets[0][1:-1].replace("'", "").split(", "))]
#             workout_enumerated = [(index + 1, value) for index, (key, value) in enumerate(workout.items())]

#             # Render the template with enumerated lists
#             return render_template('index.html', 
#                                 predicted_disease=predicted_disease,
#                                 describe=describe,
#                                 precaution=precaution_enumerated,
#                                 medication=medication_enumerated,
#                                 diets=diet_enumerated,
#                                 workout=workout_enumerated)
#         except ValueError:

#             flash("Disease not found or incorrect spelling. Please try again.", "is-danger")
#             return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        user_symptoms = [s.strip() for s in symptoms.split(',') if s.strip() in symptoms_dict]
        if not user_symptoms:
            flash("No valid symptoms found. Please try again.", "is-danger")
            return redirect(url_for('home'))

        try:
            predicted_disease = get_predicted_value(user_symptoms)
            describe, precaution, medication, diets, workout = helper(predicted_disease)

            # Enumerate the results for display
            precaution_enumerated = list(enumerate(precaution, 1))
            # list(enumerate(precaution, 1))
            medication_enumerated = [(index + 1, med) for index, med in enumerate(medication[0][1:-1].replace("'", "").split(", "))]
            # list(enumerate(medication, 1))
            diet_enumerated = [(index + 1, diet) for index, diet in enumerate(diets[0][1:-1].replace("'", "").split(", "))]
            # list(enumerate(diets, 1))
            workout_enumerated = [(index + 1, value) for index, (key, value) in enumerate(workout.items())]
            # list(enumerate(workout, 1))

            return render_template(
                'index.html',
                predicted_disease=predicted_disease,
                describe=describe,
                precaution=precaution_enumerated,
                medication=medication_enumerated,
                diets=diet_enumerated,
                workout=workout_enumerated
            )
        except KeyError as e:
            flash(str(e), "is-danger")
            return redirect(url_for('home'))









if __name__ == '__main__':
    app.run(debug=True)
