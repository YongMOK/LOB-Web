from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import Market, Dataset, MLModel, CustomUser

class MarketForm(forms.ModelForm):
    class Meta:
        model = Market
        fields = ['name']

class DatasetForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['training_file', 'date']
        widgets = {
            'date': forms.DateInput(attrs={'type': 'date'}),
        }
        
class MLModelForm(forms.ModelForm):
    class Meta:
        model = MLModel
        fields = ['name']
class CustomUserCreationForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        model = CustomUser
        fields = ('email', 'first_name', 'last_name')
        
class UploadFileForm(forms.Form):
    market_name = forms.ModelChoiceField(queryset= Market.objects.all(), label = "Market")
    model_name = forms.ModelChoiceField(queryset= MLModel.objects.all(), label = "Model")
    k_value = forms.IntegerField(label= "K Value", min_value=1, max_value=100)
    file = forms.FileField(required=False)
    dataset_id = forms.ModelChoiceField(queryset= Dataset.objects.all(), label = "Dataset", required=False)
    
class TrainModelForm(forms.Form):
    market_name = forms.ModelChoiceField(queryset=Market.objects.all(), label = "Market")
    model_name = forms.ModelChoiceField(queryset = MLModel.objects.all(), label="Model")
    dataset_id = forms.ModelChoiceField(queryset=Dataset.objects.all(),label="Dataset")
    k_value = forms.IntegerField(label="K Value", min_value=1, max_value=100)
    
class PredictModelForm(forms.Form):
    market_name = forms.ModelChoiceField(queryset=Market.objects.all(), label="Market")
    model_name = forms.ModelChoiceField(queryset=MLModel.objects.all(), label="Model")
    dataset_id = forms.ModelChoiceField(queryset=Dataset.objects.all(), label="Dataset")
    k_value = forms.IntegerField(label="K Value", min_value=1, max_value=100)
    
