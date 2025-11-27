import os
import joblib
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd 

# ───────────────────────────────────────────────
#   FONCTIONS UTILES
# ───────────────────────────────────────────────

def load_models(name):
    """
    Charger n'importe quel modèle sauvegardé dans le dossier models_ai.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models_ai')
    model_path = os.path.join(models_dir, name)
    ml_model = joblib.load(model_path)
    return ml_model

# ───────────────────────────────────────────────
#   VUES DES PAGES STATIQUES
# ───────────────────────────────────────────────

def index(request):
    return render(request, 'index.html')

def reglog_details(request):
    return render(request, 'reglog_details.html')

def reglog_atelier(request):
    return render(request, 'reglog_atelier.html')

def reglog_tester(request):
    return render(request, 'vehicles_from.html')

def xgb_details(request):
    return render(request, 'xgb_details.html')

def xgb_result(request):
    return render(request, 'xgb_result.html')

def xgb_tester(request):
    return render(request, 'xgb_tester.html')

def xgb_demonstration(request):
    return render(request, 'xgb_demonstration.html')

# ───────────────────────────────────────────────
#   LOGISTIC REGRESSION - PREDICTION VEHICULE
# ───────────────────────────────────────────────

def regLog_prediction(request):
    """
    Prédiction du type de véhicule (Camion / Touristique) selon hauteur et nombre de roues.
    """
    if request.method == 'POST':
        hauteur = float(request.POST.get('hauteur'))
        nbr_roues = float(request.POST.get('Nombre_de_roues'))
        
        model = load_models('logreg_model.pkl')
        prediction = model.predict([[hauteur, nbr_roues]])
        predicted_class = prediction[0]
        
        # Mapping des classes vers labels et images
        type_vehicules = {0: 'Camion', 1: 'Touristique'}
        img_url = {'Camion': 'images/camion.jpg', 'Touristique': 'images/touristique.jpg'}
        pred_vehicule = type_vehicules[predicted_class]
        pred_img = img_url[pred_vehicule]
        
        input_data = {'hauteur': hauteur, 'nbr_roues': nbr_roues}
        
        context = {
            'type_vehicule': pred_vehicule,
            'img_vehicule': pred_img,
            'initial_data': input_data
        }
        
        return render(request, 'reglog_results.html', context)
    
    return render(request, 'vehicles_from.html')

# ───────────────────────────────────────────────
#   XGBOOST - PREDICTION MANUELLE
# ───────────────────────────────────────────────

MODEL_PATH_XGB_FITNESS = os.path.join('models_ai', 'modele_fitness.pkl')

def xgb_prediction(request):
    """
    Prédiction individuelle à partir d’un formulaire pour XGBoost.
    """
    if request.method == 'POST':
        # Récupération des inputs
        age = float(request.POST.get('age'))
        height_cm = float(request.POST.get('height_cm'))
        weight_kg = float(request.POST.get('weight_kg'))
        heart_rate = float(request.POST.get('heart_rate'))
        blood_pressure = float(request.POST.get('blood_pressure'))
        sleep_hours = float(request.POST.get('sleep_hours'))
        nutrition_quality = float(request.POST.get('nutrition_quality'))
        activity_index = float(request.POST.get('activity_index'))
        smokes = request.POST.get('smokes')
        gender = request.POST.get('gender')

        # Encodage simple
        smokes_num = 1 if smokes.lower() == 'yes' else 0
        gender_num = 1 if gender.upper() == 'F' else 0

        input_features = [[
            age, height_cm, weight_kg, heart_rate, blood_pressure,
            sleep_hours, nutrition_quality, activity_index,
            smokes_num, gender_num
        ]]

        # Chargement du modèle et du label encoder
        package = joblib.load(MODEL_PATH_XGB_FITNESS)
        modele = package['modele']
        label_encoder = package['label_encoder_classe']

        # Prédiction
        pred = modele.predict(input_features)[0]
        pred_label = label_encoder.inverse_transform([pred])[0]

        context = {
            'prediction': pred_label,
            'initial_data': {
                'Age': age, 'Height (cm)': height_cm, 'Weight (kg)': weight_kg,
                'Heart Rate': heart_rate, 'Blood Pressure': blood_pressure,
                'Sleep Hours': sleep_hours, 'Nutrition Quality': nutrition_quality,
                'Activity Index': activity_index, 'Smokes': smokes, 'Gender': gender
            }
        }

        return render(request, 'xgb_result.html', context)

    return render(request, 'xgb_tester.html')

# ───────────────────────────────────────────────
#   XGBOOST - PREDICTION À PARTIR D’UN CSV
# ───────────────────────────────────────────────

def xgb_prediction_csv(request):
    if request.method == 'POST' and request.FILES.get('fichier'):
        fichier = request.FILES['fichier']
        df = pd.read_csv(fichier)

        # Colonnes attendues
        required_cols = [
            'age', 'height_cm', 'weight_kg', 'heart_rate', 'blood_pressure',
            'sleep_hours', 'nutrition_quality', 'activity_index', 'smokes', 'gender'
        ]

        # Vérification des colonnes
        for col in required_cols:
            if col not in df.columns:
                return render(request, 'xgb_tester.html', {'error': f"Colonne manquante : {col}"})

        # Encodage
        df['smokes'] = df['smokes'].map({'yes': 1, 'no': 0})
        df['gender'] = df['gender'].map({'F': 1, 'M': 0})

        # Chargement du modèle
        package = joblib.load(MODEL_PATH_XGB_FITNESS)
        modele = package['modele']
        label_encoder = package['label_encoder_classe']

        # Prédiction
        y_pred = modele.predict(df[required_cols])
        df['Prediction'] = label_encoder.inverse_transform(y_pred)

        # Sauvegarde session pour téléchargement
        request.session['csv_file'] = df.to_csv(index=False)

        return render(request, 'xgb_result.html', {'predictions_batch': df.to_dict(orient='records')})

    return render(request, 'xgb_tester.html')


def xgb_download_csv(request):
    """
    Générer un CSV vide pour téléchargement.
    """
    df = pd.DataFrame({'feature1': [], 'feature2': [], 'feature3': []})
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="xgb_data.csv"'
    df.to_csv(path_or_buf=response, index=False)
    return response

# ───────────────────────────────────────────────
#   REGRESSION CALORIES - FORMULAIRE
# ───────────────────────────────────────────────

MODEL_PATH_CALORIES = "models_ai/xgb_calories_regression.pkl"

def reg_prediction(request):
    """
    Prédiction individuelle de calories à partir d’un formulaire.
    """
    if request.method == 'POST':
        gender = request.POST.get('Gender')
        age = float(request.POST.get('Age'))
        height = float(request.POST.get('Height'))
        weight = float(request.POST.get('Weight'))
        duration = float(request.POST.get('Duration'))
        heart_rate = float(request.POST.get('Heart_Rate'))
        body_temp = float(request.POST.get('Body_Temp'))

        # Chargement du modèle
        package = joblib.load(MODEL_PATH_CALORIES)
        modele = package['modele']
        encoders = package['encoders']
        colonnes_features = package['colonnes_features']

        # Préparer features et encoder Gender
        df_input = pd.DataFrame([[gender, age, height, weight, duration, heart_rate, body_temp]],
                                columns=colonnes_features)
        df_input['Gender'] = encoders['Gender'].transform(df_input['Gender'])

        # Prédiction
        pred = modele.predict(df_input)[0]

        context = {
            'prediction': round(pred, 2),
            'initial_data': {
                'Gender': gender, 'Age': age, 'Height': height, 'Weight': weight,
                'Duration': duration, 'Heart_Rate': heart_rate, 'Body_Temp': body_temp
            }
        }

        return render(request, 'reg_result.html', context)

    return render(request, 'reg_tester.html')

# ───────────────────────────────────────────────
#   REGRESSION CALORIES - CSV
# ───────────────────────────────────────────────

def reg_prediction_csv(request):
    """
    Prédiction de calories à partir d’un fichier CSV.
    """
    if request.method == 'POST' and request.FILES.get('fichier'):
        fichier = request.FILES['fichier']
        df = pd.read_csv(fichier)

        package = joblib.load(MODEL_PATH_CALORIES)
        modele = package['modele']
        encoders = package['encoders']
        colonnes_features = package['colonnes_features']

        # Encoder Gender
        df['Gender'] = encoders['Gender'].transform(df['Gender'])

        # Prédiction
        df['Calories_Pred'] = modele.predict(df[colonnes_features])

        # Sauvegarde session pour téléchargement
        request.session['csv_file'] = df.to_csv(index=False)

        return render(request, 'reg_result.html', {'predictions_batch': df.to_dict(orient='records')})

    return render(request, 'reg_tester.html')


def reg_download_csv(request):
    """
    Télécharger le CSV des prédictions.
    """
    csv_file = request.session.get('csv_file')
    if not csv_file:
        return HttpResponse("Aucun fichier à télécharger.", content_type="text/plain")

    response = HttpResponse(csv_file, content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="predictions_calories.csv"'
    return response
