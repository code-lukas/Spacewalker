"""
URL configuration for backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from backend.gui.views import guiView, configurationView, MinIOWebhook, ProjectSelectionView, ExportView,\
    InferenceSettingsView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("gui/", guiView.as_view(), name="gui"),
    path("", configurationView.as_view(), name="config"),
    path("MinIOWebhook", MinIOWebhook.as_view(), name="MinIOWebhook"),
    path("projects/", ProjectSelectionView.as_view(), name="project_selection"),
    path("export/", ExportView.as_view(), name="export"),
    path("inference/", InferenceSettingsView.as_view(), name="inference"),
]
