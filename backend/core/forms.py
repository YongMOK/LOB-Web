from django import forms
from django.contrib.auth.forms import UserCreationForm, PasswordChangeForm
from .models import Market, Dataset, MLModel, CustomUser

class MarketForm(forms.ModelForm):
    class Meta:
        model = Market
        fields = ['market_name']

class DatasetForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['training_file', 'date']
        widgets = {
            'date': forms.DateInput(attrs={'type': 'date'}),
        }
        
class CustomUserCreationForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        model = CustomUser
        fields = ('email', 'first_name', 'last_name')
        
class UserSettingsForm(forms.ModelForm):
    class Meta:
        model = CustomUser
        fields = ['email', 'first_name', 'last_name']
class ChangePasswordForm(PasswordChangeForm):
    pass
    
class PredictModelForm(forms.Form):
    market_name = forms.ModelChoiceField(queryset=Market.objects.all(), label="Market")
    model_name = forms.ModelChoiceField(queryset=MLModel.objects.all(), label="Model")
    dataset_id = forms.ModelChoiceField(queryset=Dataset.objects.all(), label="Dataset")
    k_value = forms.IntegerField(label="K Value", min_value=1, max_value=100)
    

