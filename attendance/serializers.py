from rest_framework import serializers
from .models import Attendance
from admin_panel.models import Employee

# Employee Serializer
class EmployeeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Employee
        fields = ['id', 'user', 'photo']

# Attendance Serializer
class AttendanceSerializer(serializers.ModelSerializer):
    employee = EmployeeSerializer()

    class Meta:
        model = Attendance
        fields = ['employee', 'date', 'time_in', 'time_out']


class AttendanceCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Attendance
        fields = ['employee', 'date', 'time_in', 'time_out']
