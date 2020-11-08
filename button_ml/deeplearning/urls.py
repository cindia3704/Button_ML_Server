from django.urls import path, include
from .views import post_cloth
from . import views
# , change_cloth, delete_cloth, get_set

urlpatterns = [
    path('postCloth/', views.post_cloth),
    # path('changeCloth/', views.change_cloth),
    path('deleteCloth/', views.delete_cloth),
    # path('getSet/', views.get_set),
]
