from django.contrib import admin
from .models import Employee, Attendance

from django.contrib import admin
from .models import Employee

@admin.register(Employee)
class EmployeeAdmin(admin.ModelAdmin):
    # Fields to display in the admin list view
    list_display = ('id', 'user', 'created_at')
    
    # Fields to use for searching in the admin interface
    search_fields = ('user__username', 'user__email')
    
    # Fields to filter by in the admin interface
    list_filter = ('created_at',)
    
    # Read-only fields (optional, for auditing purposes)
    readonly_fields = ('created_at',)

@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ('employee', 'date', 'time_in', 'time_out')
    search_fields = ('employee__username', 'date')
    list_filter = ('date',)
    list_per_page = 20
