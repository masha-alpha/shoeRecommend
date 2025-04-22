# your_project/urls.py

from django.contrib import admin
from django.urls import path, include  # <-- include is important

urlpatterns = [
    path('admin/', admin.site.urls),
    path('shoecommend/', include('recommendation.urls')),  # <-- this links your app's urls
]
