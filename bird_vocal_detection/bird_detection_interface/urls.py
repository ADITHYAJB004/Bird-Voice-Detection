from django.urls import path
from . import views

urlpatterns = [
    path('detect_voice/', views.process_bird_voice, name='home'),
    path('result/', views.process_bird_voice, name='result'),  # Result page
]