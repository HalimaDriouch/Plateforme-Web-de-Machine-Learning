from django.urls import path
from . import views

urlpatterns = [
    # Page d'accueil
    path('', views.home, name='home'),
    
    # Pages de détails (Statiques)
    path('details/regression-logistique/', views.regLog_details, name='regLog_details'),
    
    path('tester/<str:algo_name>/', views.tester_modele, name='tester_modele'),
    path('details/dt-classification/', views.dt_class_details, name='dt_class_details'),
    path('details/rf-classification/', views.rf_class_details, name='rf_class_details'),
    path('details/svm/', views.svm_class_details, name='svm_details'),
    path('details/linear-regression/', views.lin_reg_details, name='lin_reg_details'),
    path('details/rf-regression/', views.rf_reg_details, name='rf_reg_details'),
    path('details/dt-regression/', views.dt_reg_details, name='dt_reg_details'),
    path('details/svm-regression/', views.svm_reg_details, name='svm_reg_details'),
    path('details/XGboost-classification/', views.xgb_classi_details, name='XGboost_class_details'),
    path('details/XGboost-regression/', views.xgb_reg_details, name='XGboost_reg_details'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('about/', views.about, name='about'),
    path('historique/', views.historique_view, name='historique'),
    path('historique/clear/', views.clear_history, name='clear_history'),
    path('algo_regression/', views.regression, name='regression'),
    path('algo_classification/', views.classification, name='classification'),
]
