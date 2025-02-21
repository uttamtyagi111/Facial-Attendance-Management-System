from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Employee, Attendance
from .serializers import AttendanceSerializer
from face_recognition.recognition import recognize_face
from datetime import datetime, timedelta


def get_image_data(request):
    """Utility to retrieve image data from the request."""
    image_file = request.FILES.get('photo')
    if not image_file:
        return None, Response({"error": "No photo provided."}, status=status.HTTP_400_BAD_REQUEST)
    return image_file.read(), None


def get_recognized_employee(image_data):
    """Utility to recognize face and retrieve employee."""
    recognized_employee_id = recognize_face(image_data)
    if not recognized_employee_id:
        return None, Response({"error": "Face not recognized."}, status=status.HTTP_400_BAD_REQUEST)
    try:
        return Employee.objects.get(id=recognized_employee_id), None
    except Employee.DoesNotExist:
        return None, Response({"error": "Employee not found."}, status=status.HTTP_404_NOT_FOUND)


class MarkInAttendance(APIView):
    def post(self, request):
        image_data, error_response = get_image_data(request)
        if error_response:
            return error_response

        employee, error_response = get_recognized_employee(image_data)
        if error_response:
            return error_response

        today = datetime.now().date()
        attendance, created = Attendance.objects.get_or_create(
            employee=employee,
            date=today,
            defaults={'time_in': datetime.now().time()}
        )

        message = "Marked time-in successfully." if created else "Time-in already marked for today."
        serializer = AttendanceSerializer(attendance)
        return Response({"message": message, "attendance": serializer.data}, status=status.HTTP_200_OK)


class MarkOutAttendance(APIView):
    def post(self, request):
        image_data, error_response = get_image_data(request)
        if error_response:
            return error_response

        employee, error_response = get_recognized_employee(image_data)
        if error_response:
            return error_response

        today = datetime.now().date()

        try:
            attendance = Attendance.objects.get(employee=employee, date=today)

            if attendance.time_out:
                return Response({"message": "Time-out already marked for today."}, status=status.HTTP_400_BAD_REQUEST)

            time_in_datetime = datetime.combine(today, attendance.time_in)
            min_time_out = time_in_datetime + timedelta(hours=9)

            if datetime.now() < min_time_out:
                remaining_time = min_time_out - datetime.now()
                hours, remainder = divmod(remaining_time.seconds, 3600)
                minutes, _ = divmod(remainder, 60)
                return Response({
                    "error": "You cannot mark time-out yet.",
                    "message": f"Please wait {hours} hours and {minutes} minutes."
                }, status=status.HTTP_400_BAD_REQUEST)

            attendance.time_out = datetime.now().time()
            attendance.save()
            serializer = AttendanceSerializer(attendance)
            return Response({"message": "Marked time-out successfully.", "attendance": serializer.data},
                            status=status.HTTP_200_OK)

        except Attendance.DoesNotExist:
            return Response({"error": "Time-in not marked for today. Cannot mark time-out."},
                            status=status.HTTP_400_BAD_REQUEST)

