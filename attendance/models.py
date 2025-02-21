from django.db import models
from admin_panel.models import Employee
from django.contrib.auth.models import User


class Attendance(models.Model):
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    date = models.DateField()
    time_in = models.TimeField()
    time_out = models.TimeField(null=True, blank=True)
    status = models.CharField(max_length=10, choices=[('Present', 'Present'), ('Absent', 'Absent')], default='Present')



    def __str__(self):
        return f"{self.employee.user.username} - {self.date} - {self.status}"
    
class Employee(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    # Other fields
    created_at = models.DateTimeField(auto_now_add=True)
 
    def __str__(self):
        return self.user.username