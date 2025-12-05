from django.db import models
from django.contrib.auth.models import User

class Historique(models.Model):
    # Lien avec l'utilisateur (Si l'utilisateur est supprimé, son historique aussi)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # Nom de l'algorithme utilisé (ex: "XGBoost Regression")
    algo_name = models.CharField(max_length=100)
    
    # Les données saisies (Age, Poids, etc.) stockées sous forme de texte ou JSON
    # On utilise JSON pour être flexible (car les champs changent selon l'algo)
    input_data = models.JSONField() 
    
    # Le résultat (ex: "FIT" ou "450.5 kCal")
    resultat = models.CharField(max_length=100)
    
    # La date de la prédiction (automatique)
    date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.algo_name} - {self.date}"
    
    # Pour afficher les plus récents en premier
    class Meta:
        ordering = ['-date']
