from django.shortcuts import render
from attendance.models import Employee, Attendance
from django.http import HttpResponse
from django.http import JsonResponse, HttpResponseRedirect


def dashboard(request):
    if not request.user.is_superuser:
        return HttpResponse("Unauthorized", status=403)

    employees = Employee.objects.all()
    context = {'employees': employees}
    return render(request, 'admin_panel/dashboard.html', context)




from django.shortcuts import render
from django.http import JsonResponse
from users.models import Employee
from attendance.models import Attendance
from django.views import View

class AdminDashboard(View):
    def get(self, request):
        # Render admin dashboard
        employees = Employee.objects.all()
        attendance_data = Attendance.objects.all()
        return render(request, 'admin_dashboard.html', {'employees': employees, 'attendance_data': attendance_data})

class ManageEmployee(View):
    def post(self, request):
        # Logic for adding a new employee
        pass
