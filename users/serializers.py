from rest_framework import serializers
from .models import Employee
from django.contrib.auth.hashers import make_password


class UserRegistrationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Employee
        fields = ['username', 'email', 'password', 'photo']

    def create(self, validated_data):
        # Hash password before saving
        password = validated_data.pop('password')
        validated_data['password'] = make_password(password)
        return super().create(validated_data)
