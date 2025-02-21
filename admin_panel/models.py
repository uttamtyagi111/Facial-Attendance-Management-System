from django.contrib.auth.models import User
from django.db import models

# Extend the default User model with admin-specific features if needed.
class AdminProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    is_super_admin = models.BooleanField(default=False)

    def __str__(self):
        return self.user.username




from django.db import models

class Employee(models.Model):
    name = models.CharField(max_length=100)
    photo = models.ImageField(upload_to='employee_photos/')
    # other fields for the Employee model
    
    def __str__(self):
        return self.name
