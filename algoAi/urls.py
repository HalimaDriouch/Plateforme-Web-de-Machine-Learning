from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('reglog_details/', views.reglog_details, name='reglog_details'),
    path('reglog_atelier/', views.reglog_atelier, name='reglog_atelier'),
    path('reglog_tester/', views.reglog_tester, name='reglog_tester'),
    path('reglog_prediction', views.regLog_prediction, name='reglog_prediction'),
]