from django.urls import path
from .views import MarkInAttendance, MarkOutAttendance

urlpatterns = [
    path('mark_in/', MarkInAttendance.as_view(), name='mark_in'),
    path('mark_out/', MarkOutAttendance.as_view(), name='mark_out'),
]
