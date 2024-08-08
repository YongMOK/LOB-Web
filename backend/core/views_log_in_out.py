from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login,logout
from django.views.decorators.http import require_http_methods
from .forms import CustomUserCreationForm
from django.contrib import messages

@require_http_methods(["GET", "POST"])
def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = authenticate(request, username=email, password=password) 
        print(user)
        if user is not None:
            login(request, user)
            request.session['is_admin'] = user.is_superuser
            messages.success(request, f'Welcome back, user!')
            return redirect('home')  
        else:
            messages.error(request, 'Invalid Email or Password. Please try again.')
            return render(request, 'Login_out/login.html')  
    else:
        return render(request, 'Login_out/login.html')  
    


def register(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_staff = True
            user.save()
            messages.success(request, 'Registration successful! Please log in.')
            return redirect('login')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field}: {error}")
    else:
        form = CustomUserCreationForm()
    return render(request, 'Login_out/register.html', {'form': form})

@require_http_methods(["POST"])
def logout_view(request):
    # Log the user out
    logout(request)
    # Redirect to the login page
    return redirect('login')
