from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('reglog_details/', views.reglog_details, name='reglog_details'),
    path('reglog_atelier/', views.reglog_atelier, name='reglog_atelier'),
    path('reglog_tester/', views.reglog_tester, name='reglog_tester'),
    path('reglog_prediction', views.regLog_prediction, name='reglog_prediction'),
    path('xgb_prediction', views.xgb_prediction, name='xgb_prediction'),
    path('xgb/demonstration/', views.xgb_demonstration, name='xgb_demonstration'),
    path('xgb_details/', views.xgb_details, name='xgb_details'),
    path('xgb_tester/', views.xgb_tester, name='xgb_tester'),
    path('xgb-prediction-csv/', views.xgb_prediction_csv, name='xgb_prediction_csv'),
    path('xgb-download-csv/', views.xgb_download_csv, name='xgb_download_csv'),
]