from django.core.validators import FileExtensionValidator
from django.utils import timezone
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from django.contrib.auth.models import PermissionsMixin
from django.db import models
from datetime import time

class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')

        return self.create_user(email, password, **extra_fields)

class CustomUser(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(unique=True)
    first_name = models.CharField(max_length=30, blank=True)
    last_name = models.CharField(max_length=30, blank=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)

    objects = CustomUserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    def __str__(self):
        return self.email

    # Add these methods if not using PermissionsMixin
    def has_perm(self, perm, obj=None):
        "Does the user have a specific permission?"
        return True  # Simplest possible answer: Yes, always

    def has_module_perms(self, app_label):
        "Does the user have permissions to view the app `app_label`?"
        return True  # Simplest possible answer: Yes, always

class Market(models.Model):
    name = models.CharField(max_length=50, unique=True)
    start_time = models.TimeField(auto_now_add=False, blank=False, default=time(10, 30))
    end_time = models.TimeField(auto_now_add=False, blank=False, default=time(18, 0))
    
    def __str__(self):
        return f"Market : {self.name}, open time: {self.start_time}, end time: {self.end_time}"

class Dataset(models.Model):
    market = models.ForeignKey(Market, related_name='datasets', on_delete=models.CASCADE)
    training_file = models.FileField(upload_to="datasets/", validators=[FileExtensionValidator(['csv', 'txt'])])
    date = models.DateField(default=timezone.now)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.market.name} - {self.training_file.name} - {self.date} - uploaded at {self.uploaded_at}"

class ProcessedDataset(models.Model):
    dataset = models.ForeignKey(Dataset, related_name='processed_versions', on_delete=models.CASCADE)
    cleaned_file = models.FileField(upload_to="processed_datasets/cleaned/", validators=[FileExtensionValidator(['csv', 'txt'])])
    normalized_file = models.FileField(upload_to="processed_datasets/normalized/", validators=[FileExtensionValidator(['csv', 'txt'])])
    processed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.dataset.market.name} - {self.dataset.date} - Processed"

class MLModel(models.Model):
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return f"{self.name} model"
class ModelParameter(models.Model):
    model = models.ForeignKey(MLModel, related_name='parameters', on_delete=models.CASCADE)
    processed_dataset = models.ForeignKey(ProcessedDataset, related_name='parameters', on_delete=models.CASCADE)
    parameters = models.JSONField(blank=True, null=True)  # Keeping this in case you still need it
    model_file = models.FileField(upload_to="trained_models/", validators=[FileExtensionValidator(['joblib'])], blank=True, null=True)
    trained_at = models.DateTimeField(auto_now_add=True)
    k = models.IntegerField()

    def __str__(self):
        return f"{self.model.name} Parameters of dataset at- {self.processed_dataset.dataset.date} - trained at - {self.trained_at}"

class Prediction(models.Model):
    model = models.ForeignKey(MLModel, related_name='predictions', on_delete=models.CASCADE)
    processed_dataset = models.ForeignKey(ProcessedDataset, related_name='predictions', on_delete=models.CASCADE)
    predictions = models.JSONField() 
    predicted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.model.name} - {self.processed_dataset.dataset.date} -at {self.predicted_at} Predictions"

class Evaluation(models.Model):
    prediction = models.ForeignKey(Prediction, related_name='evaluations', on_delete=models.CASCADE)
    accuracy = models.FloatField()
    classification_report = models.JSONField()
    confusion_matrix = models.JSONField()
    evaluated_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.prediction.model.name} - {self.prediction.processed_dataset.dataset.date} - Evaluation"
    
class BestModel(models.Model):
    market = models.ForeignKey(Market, related_name='best_models', on_delete=models.CASCADE)
    model = models.ForeignKey(MLModel, related_name='best_models', on_delete=models.CASCADE)
    evaluation = models.ForeignKey(Evaluation, related_name='best_models', on_delete=models.CASCADE)
    best_k = models.IntegerField()
    save_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Best model for {self.market.name} - {self.model.name} selected at {self.save_at}"
    

class Results_Client(models.Model):
    market = models.ForeignKey(Market, related_name='result_client', on_delete=models.CASCADE)
    result = models.JSONField()
    upload_at = models.DateTimeField(auto_now_add=True)
