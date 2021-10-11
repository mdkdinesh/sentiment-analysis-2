from django.urls import path
from . import views

urlpatterns = [
    path("", views.welcome),
    path("index/", views.index),
    path('dataset/', views.dataset),
    path('city/', views.city),
    path('state/', views.state),
    path("chart/", views.charts_sample),
    path("freq/", views.freq),
    path('total_dataset/', views.total_dataset),
    path('all_dataset/', views.all_dataset),
    path('pie_data/', views.pie_data),
    path('welcome/', views.welcome),
    path('positive/', views.positive),
    path('neutral/', views.neutral),
    path('negative/', views.negative),
    path('star_rating/', views.star_rating),
    path('Top_Positive_States/', views.Top_Positive_States),
    path('Top_Negative_States/', views.Top_Negative_States),
    path('Top_Positive_City/', views.Top_Positive_City),
    path('Top_Negative_City/', views.Top_Negative_City),
    path('Top_Neutral_States/', views.Top_Neutral_States),
    path('Top_Neutral_City/', views.Top_Neutral_City),
    path('Date_Based_reviews/', views.Date_Based_reviews),
    path('Top_positive_keywords/', views.Top_positive_keywords),
    path('Top_negative_keywords/', views.Top_negative_keywords),
]
