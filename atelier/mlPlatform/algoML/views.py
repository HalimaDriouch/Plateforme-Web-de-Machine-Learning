import os
import joblib
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.conf import settings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.shortcuts import redirect
from .models import Historique
import json
# --- 1. CONFIGURATION DES MODÈLES ---
# Mise à jour des chemins pour inclure les sous-dossiers
MODELS_CONFIG = {
    # --- CLASSIFICATION ---
    'log_class': {
        'file': os.path.join('classification', 'logistic_reg.pkl'), 
        'type': 'fitness', 
        'name': 'Régression Logistique'
    },
    'dt_class': {
        'file': os.path.join('classification', 'decision_tree_classification.pkl'), 
        'type': 'fitness', 
        'name': 'Arbre de Décision'
    },
    ## hna ba9i anzido les modeles li khassin 
    ## remplacer lpath dyal kola modele f file : os.path .....()
    'rf_class': {
        'file': os.path.join('classification', 'rf_class.pkl'), 
        'type': 'fitness', 
        'name': 'Random Forest Classifier'
    },
    'svm_class': {
        'file': os.path.join('classification', 'svm_class.pkl'), 
        'type': 'fitness', 
        'name': 'SVM Classifier'
    },
    'xgb_class': {
        'file': os.path.join('classification', 'xgb_class.pkl'), 
        'type': 'fitness', 
        'name': 'XGboost Classifier'
    },
    
    # --- REGRESSION ---
    'lin_reg': {
        'file': os.path.join('regression', 'linear_reg.pkl'), 
        'type': 'calories', 
        'name': 'Régression Linéaire'
    },
    'dt_reg': {
        'file': os.path.join('regression', 'dt_reg_opt.pkl'), 
        'type': 'calories', 
        'name': 'Arbre de Décision (Rég)'
    },
    'rf_reg': {
        'file': os.path.join('regression', 'rf_regression.pkl'), 
        'type': 'calories', 
        'name': 'Random Forest (Rég)'
    },
    'svm_reg': {
        'file': os.path.join('regression', 'svr_reg.pkl'), 
        'type': 'calories', 
        'name': 'SVM Regression'
    },
    'xgboost_reg': {
        'file': os.path.join('regression', 'xgb_reg.pkl'), 
        'type': 'calories', 
        'name': 'XGboost Regression'
    },
}
# --- 2. MOTEUR IA (SINGLETON) ---
AI_ENGINE = {'models': {}, 'scalers': {}}
def init_engine():
    """Charge les modèles .pkl et reconstruit les scalers une seule fois au démarrage."""
    if AI_ENGINE['models']: return
    print("⚡ Initialisation du moteur IA pour algoML...")
    
    # Chemin racine des modèles : mlPlatform/algoML/ml_models/
    base_models_dir = os.path.join(settings.BASE_DIR, 'algoML', 'ml_models')
    # A. SCALER FITNESS (Reconstruit depuis le CSV racine)
    try:
        csv_path = os.path.join(settings.BASE_DIR, 'fitness_dataset.csv')
        df_c = pd.read_csv(csv_path)
        
        # Nettoyage identique à l'entraînement
        df_c['smokes'] = df_c['smokes'].astype(str).map({'yes': 1, 'no': 0, '1': 1, '0': 0}).fillna(0)
        le_gen = LabelEncoder()
        df_c['gender'] = le_gen.fit_transform(df_c['gender'].astype(str))
        df_c = df_c.fillna(df_c.mean(numeric_only=True))
        
        scaler_c = StandardScaler()
        scaler_c.fit(df_c.drop(columns=['is_fit']))
        AI_ENGINE['scalers']['fitness'] = scaler_c
        print("✅ Scaler Fitness prêt.")
    except Exception as e: 
        print(f"⚠️ Erreur Scaler Fitness: {e}")
    # B. SCALER CALORIES (Reconstruit depuis le CSV racine)
    try:
        csv_path = os.path.join(settings.BASE_DIR, 'calories_pred.csv')
        df_r = pd.read_csv(csv_path)
        
        le_gen_r = LabelEncoder()
        df_r['Gender'] = le_gen_r.fit_transform(df_r['Gender'])
        
        
        # 2. FORCEZ L'ORDRE EXACT QUE VOUS AVEZ TROUVÉ
        ordre_officiel = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
        
        # On filtre et on réordonne le DataFrame d'un coup
        # Cela supprime aussi automatiquement User_ID et Calories car ils ne sont pas dans la liste
        df_finale = df_r[ordre_officiel]
        
        scaler_r = StandardScaler()
        scaler_r.fit(df_finale)
        AI_ENGINE['scalers']['calories'] = scaler_r
        print("✅ Scaler Calories prêt.")
    except Exception as e: 
        print(f"⚠️ Erreur Scaler Calories: {e}")
    # C. CHARGEMENT DES PKL
    for key, config in MODELS_CONFIG.items():
        try:
            # On combine le dossier de base avec le chemin relatif (ex: classification/file.pkl)
            path = os.path.join(base_models_dir, config['file'])
            
            if os.path.exists(path):
                AI_ENGINE['models'][key] = joblib.load(path)
                print(f"  -> Chargé: {config['name']}")
            else:
                print(f"  ❌ Manquant: {path}")
        except Exception as e:
            print(f"  ❌ Erreur {key}: {e}")
# Lancer l'initialisation au démarrage
init_engine()
# --- 3. VUES DJANGO ---
def regression(request):
    """Page d'accueil Dashboard"""
    return render(request, 'regression.html')

def classification(request):
    """Page d'accueil Dashboard"""
    return render(request, 'classification.html')

def regLog_details(request):
    """Page de détails statique"""
    return render(request, 'regLog_details.html')
def lin_reg_details(request):
    return render(request, 'lin_reg_details.html')
def dt_reg_details(request):
    return render(request, 'dt_reg_details.html')
def dt_class_details(request):
    return render(request, 'dt_classi_details.html')
def rf_class_details(request):
    return render(request, 'rf_classi_details.html')
def rf_reg_details(request):
    return render(request, 'rf_reg_details.html')
def svm_class_details(request):
    return render(request, 'svm_classi_details.html')
def svm_reg_details(request):
    return render(request, 'svm_reg_details.html')
def xgb_classi_details(request):
    return render(request, 'XGboost_classi_details.html')
def xgb_reg_details(request):
    return render(request, 'XGboost_reg_details.html')
def about(request):
    return render(request, 'about.html')

def home(request):
    return render(request, 'home.html')



def tester_modele(request, algo_name):
    config = MODELS_CONFIG.get(algo_name)
    if not config:
        return render(request, 'index.html', {'error': "Modèle introuvable"})
    type_pred = config['type']
    context = {'algo_name': algo_name, 'nom_modele': config['name'], 'type_pred': type_pred}
    if request.method == 'POST':
        try:
            # 1. Récupération des données communes
            age = float(request.POST.get('age', 0))
            height = float(request.POST.get('height', 0))
            weight = float(request.POST.get('weight', 0))
            heart = float(request.POST.get('heart_rate', 0))
            gender = int(request.POST.get('gender', 0))
            
            model = AI_ENGINE['models'].get(algo_name)
            
            # Variable pour stocker le résultat texte (IMPORTANT : Doit être définie dans les 2 cas)
            res_str = "" 
            # === CAS 1 : CLASSIFICATION (FITNESS) ===
            if type_pred == 'fitness':
                bp = 120.0
                sleep = float(request.POST.get('sleep', 7))
                nutri = float(request.POST.get('nutrition', 5))
                activ = float(request.POST.get('activity', 3))
                smokes = int(request.POST.get('smokes', 0))
                
                features = [[age, height, weight, heart, bp, sleep, nutri, activ, smokes, gender]]
                
                if 'fitness' in AI_ENGINE['scalers']:
                    features = AI_ENGINE['scalers']['fitness'].transform(features)
                
                pred = model.predict(features)[0]
                
                # CALCUL DES PROBABILITÉS ET SCORE DE CERTITUDE DU MODÈLE
                try:
                    # Obtenir les probabilités de prédiction
                    probas = model.predict_proba(features)[0]
                    
                    # Score de certitude du modèle = probabilité de la classe prédite
                    confidence = round(max(probas) * 100, 2)
                    
                    # Probabilités pour chaque classe
                    prob_not_fit = round(probas[0] * 100, 2)  # Classe 0 = NOT FIT
                    prob_fit = round(probas[1] * 100, 2)      # Classe 1 = FIT
                    
                    # Nive de certitude
                    if confidence >= 90:
                        certitude = "Très Haute"
                        certitude_badge = "success"
                    elif confidence >= 75:
                        certitude = "Haute"
                        certitude_badge = "info"
                    elif confidence >= 60:
                        certitude = "Moyenne"
                        certitude_badge = "warning"
                    else:
                        certitude = "Faible"
                        certitude_badge = "danger"
                    
                    context['confiance'] = confidence
                    context['prob_fit'] = prob_fit
                    context['prob_not_fit'] = prob_not_fit
                    context['certitude'] = certitude
                    context['certitude_badge'] = certitude_badge
                    
                except Exception as e:
                    print(f"⚠️ Impossible de calculer les probabilités: {e}")
                    context['confiance'] = None
                
                # DÉFINITION DU RÉSULTAT POUR L'HISTORIQUE
                res_str = "FIT 💪" if pred == 1 else "NOT FIT ⚠️"
                context['resultat'] = res_str
            # === CAS 2 : RÉGRESSION (CALORIES) ===
            else: 
                duration = float(request.POST.get('duration', 0))
                temp = float(request.POST.get('body_temp', 37))
                
                features = [[gender, age, height, weight, duration, heart, temp]]
                
                if 'calories' in AI_ENGINE['scalers']:
                    features = AI_ENGINE['scalers']['calories'].transform(features)
                
                raw_val = model.predict(features)[0]
                val_float = float(raw_val)
                
                # DÉFINITION DU RÉSULTAT POUR L'HISTORIQUE
                res_str = f"{round(val_float, 2)} kCal brûlées 🔥"
                context['resultat'] = res_str
            
            if request.user.is_authenticated:
                donnees = request.POST.dict()
                if 'csrfmiddlewaretoken' in donnees:
                    del donnees['csrfmiddlewaretoken']
                
                Historique.objects.create(
                    user=request.user,
                    algo_name=config['name'],
                    input_data=donnees,
                    resultat=res_str  # On utilise la variable créée plus haut
                )
                print(f"✅ Historique sauvegardé : {res_str}")
            # ====================================================
        except Exception as e:
            print(f"ERREUR : {e}")
            context['erreur'] = f"Erreur : {str(e)}"
    return render(request, 'model_tester.html', context)

def signup_view(request):
    if request.method == 'POST':
        # 1. Récupérer les données du formulaire
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        password_confirm = request.POST.get('password_confirm')
        
        # 2. Vérifications de base
        if password != password_confirm:
            messages.error(request, "Les mots de passe ne correspondent pas.")
            return render(request, 'signup.html')
        
        if User.objects.filter(username=username).exists():
            messages.error(request, "Ce nom d'utilisateur est déjà pris.")
            return render(request, 'signup.html')
        # 3. Création de l'utilisateur
        try:
            user = User.objects.create_user(username=username, email=email, password=password)
            user.save()
            
            # 4. Connexion automatique après inscription
            login(request, user)
            messages.success(request, f"Bienvenue, {username} !")
            return redirect('home') # Redirige vers l'accueil
            
        except Exception as e:
            messages.error(request, f"Erreur lors de l'inscription : {e}")
    return render(request, 'signup.html')
def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        # Vérification des identifiants
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, f"Ravi de vous revoir, {username} !")
            return redirect('home')
        else:
            messages.error(request, "Nom d'utilisateur ou mot de passe incorrect.")
    
    return render(request, 'login.html')
def logout_view(request):
    logout(request)
    messages.info(request, "Vous avez été déconnecté.")
    return redirect('login')
from django.contrib.auth.decorators import login_required
@login_required # Oblige à être connecté pour voir cette page
def historique_view(request):
    # Récupère l'historique de l'utilisateur connecté uniquement
    historiques = Historique.objects.filter(user=request.user).order_by('-date')
    return render(request, 'historique.html', {'historiques': historiques})

@login_required
def clear_history(request):
    if request.method == 'POST':
        Historique.objects.filter(user=request.user).delete()
        messages.success(request, "Votre historique a été effacé avec succès.")
    return redirect('historique')