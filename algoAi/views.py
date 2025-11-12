import os
import joblib
from django.shortcuts import render

def load_models(name):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models_ai')
    model_path = os.path.join(models_dir, name)
    ml_model = joblib.load(model_path)
    return ml_model

def index(request):
    return render(request, 'index.html')

def reglog_details(request):
    return render(request, 'reglog_details.html')

def reglog_atelier(request):
    return render(request, 'reglog_atelier.html')

# ✅ AJOUTER CETTE VUE MANQUANTE
def reglog_tester(request):
    return render(request, 'vehicles_from.html')

def regLog_prediction(request):
    # Tache 1: Recevoir le Colis
    if request.method == 'POST':
        # Tache 2 : Déballer le Colis
        hauteur = float(request.POST.get('hauteur'))
        nbr_roues = float(request.POST.get('Nombre_de_roues'))
        
        # Tache 3 : Réveiller l'Expert
        # cette fonction (load_models) est défini avant
        model = load_models('logreg_model.pkl')
        
        # Tache 4 : Poser la Question à l'Expert
        prediction = model.predict([[hauteur, nbr_roues]])
        predicted_class = prediction[0]
        
        # Tache 5 : Traduire la Réponse
        type_vehicules = {0: 'Camion', 1: 'Touristique'}
        img_url = {'Camion': 'images/camion.jpg', 'Touristique': 'images/touristique.jpg'}
        pred_vehicule = type_vehicules[predicted_class]
        pred_img = img_url[pred_vehicule]
        
        # Tache 6 : Préparer le Plateau-Repas (context)
        input_data = {
            'hauteur': hauteur,
            'nbr_roues': nbr_roues
        }
        
        context = {
            'type_vehicule': pred_vehicule,
            'img_vehicule': pred_img,
            'initial_data': input_data
        }
        
        return render(request, 'reglog_results.html', context)
    
    return render(request, 'vehicles_from.html')