from django.urls import path
from .views import csv_to_html
from . import views

urlpatterns = [
    path('csv_to_html/', csv_to_html, name='csv_to_html'),
    path('regresion_lineal/', views.regresion_lineal, name='regresion_lineal'),
]