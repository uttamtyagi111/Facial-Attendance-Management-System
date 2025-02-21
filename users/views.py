from rest_framework import serializers
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from django.contrib.auth.hashers import make_password
from .models import Employee
from .serializers import UserRegistrationSerializer

class UserRegistrationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Employee
        fields = ['username', 'email', 'password', 'photo']

    def create(self, validated_data):
        # Hash the password before saving it
        password = validated_data.pop('password')
        validated_data['password'] = make_password(password)  # Correct use of make_password
        return super().create(validated_data)

class UserRegistrationView(APIView):
    def post(self, request):
        serializer = UserRegistrationSerializer(data=request.data)
        
        if serializer.is_valid():
            # Save the new user (Employee) instance
            employee = serializer.save()

            # You can add facial recognition or additional logic here if needed

            return Response({
                "message": "User registered successfully!",
                "employee": {
                    "id": employee.id,
                    "username": employee.username,
                    "email": employee.email,
                    "photo_url": employee.photo.url
                }
            }, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
