from django.contrib.auth.decorators import login_required
from django.shortcuts import render

@login_required
def home(request):
    if request.user.is_superuser:
        request.session['is_admin'] = True
    else:
        request.session['is_admin'] = False
    return render(request, 'Home/home.html')