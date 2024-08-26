
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.utils.html import format_html
from .models import (CustomUser,Market, Dataset, ProcessedDataset, MLModel, ModelParameter, 
                     Prediction, Evaluation, BestModel, Dataset_Prediction, Results_Client)

class CustomUserAdmin(UserAdmin):
    model = CustomUser
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal info', {'fields': ('first_name', 'last_name')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login',)}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'password1', 'password2', 'first_name', 'last_name', 'is_active', 'is_staff', 'is_superuser'),
        }),
    )
    list_display = ['email', 'first_name', 'last_name', 'is_active', 'is_staff', 'is_superuser']
    list_filter = ['is_staff', 'is_superuser']
    search_fields = ('email', 'first_name', 'last_name')
    ordering = ('email',)
class CustomMarketAdmin(admin.ModelAdmin):
    list_display = ['market_name', 'opening_time', 'closing_time']
    search_fields = ['market_name']
    
class CustomDatasetAdmin(admin.ModelAdmin):
    list_display = ["file_name","market_name", "date", "uploaded_at"] 
    list_filter = ['market__market_name']  # Makes the market dropdown searchable
    def market_name(self,obj):
        return obj.market.market_name
    def file_name(self,obj):
        return obj.training_file.name.split('/')[-1]
    
    file_name.short_description = 'File Name'
    market_name.short_description = 'Market Name'

class CustomProcessedDatasetAdmin(admin.ModelAdmin):
    list_display = ["dataset_name","cleaned_file_name", "normalized_file_name", "processed_at"] 
    list_filter = ["dataset__training_file"]
    def dataset_name(self,obj):
        return obj.dataset.training_file.name.split('/')[-1]
    def cleaned_file_name(self, obj):
        return obj.cleaned_file.name.split('/')[-1]
    def normalized_file_name(self, obj):
        return obj.normalized_file.name.split('/')[-1]
    
    cleaned_file_name.short_description = 'Cleaned File Name'
    normalized_file_name.short_description = 'Normalized File Name'
    dataset_name.short_description = 'Dataset Name'

class CustomMLModelAdmin(admin.ModelAdmin):
    list_display = ['model_name']
    search_fields = ['model_name']

class CustomModelParameterAdmin(admin.ModelAdmin):
    list_display = ['model_name', 'dataset_name', 'trained_at', 'k']
    list_filter = ['model__model_name', 'trained_at']

    def model_name(self, obj):
        return obj.model.model_name
    model_name.short_description = 'Model Name'

    def dataset_name(self, obj):
        return obj.processed_dataset.dataset.training_file.name.split('/')[-1]
    dataset_name.short_description = 'Dataset Name'

class CustomPredictionAdmin(admin.ModelAdmin):
    list_display = ['model_name', 'dataset_name', 'predicted_at']
    list_filter = ['model__model_name', 'processed_dataset__dataset__market__market_name', 'predicted_at']
    search_fields = ['model__model_name', 'processed_dataset__dataset__training_file']

    def model_name(self, obj):
        return obj.model.model_name
    model_name.short_description = 'Model Name'

    def dataset_name(self, obj):
        return obj.processed_dataset.dataset.training_file.name.split('/')[-1]
    dataset_name.short_description = 'Dataset Name'

class CustomEvaluationAdmin(admin.ModelAdmin):
    list_display = ['model_name', 'dataset_name', 'accuracy', 'evaluated_at']
    list_filter = ['prediction__model__model_name', 'evaluated_at']
    search_fields = ['prediction__model__model_name', 'accuracy']
    readonly_fields = ['evaluation_details']
    
    @admin.display(description='Evaluation Details')
    def evaluation_details(self, obj):
        classification_report = obj.classification_report
        confusion_matrix = obj.confusion_matrix
        
        # Format classification report with added padding
        report_html = "<h4>Classification Report : </h4><table><tr><th style='padding: 9px;'>Class</th><th style='padding: 9px;'>Precision</th><th style='padding: 9px;'>Recall</th><th style='padding: 9px;'>F1-Score</th><th style='padding: 9px;'>Support</th></tr>"
        for class_label, metrics in classification_report.items():
            if class_label == 'accuracy':
                report_html += f"<tr><td style='padding: 9px;'>{class_label}</td><td style='padding: 9px;'>{metrics:.4f}</td><td> </td><td> </td><td> </td></tr>"
            elif isinstance(metrics, dict):
                report_html += f"<tr><td style='padding: 9px;'>{class_label}</td><td style='padding: 9px;'>{metrics['precision']:.4f}</td><td style='padding: 9px;'>{metrics['recall']:.4f}</td><td style='padding: 9px;'>{metrics['f1-score']:.4f}</td><td style='padding: 9px;'>{metrics['support']}</td></tr>"
        report_html += "</table>"
        
        # Format confusion matrix with added padding
        matrix_html = "<h4>Confusion Matrix :</h4><table>"
        for row in confusion_matrix:
            matrix_html += "<tr>"
            for value in row:
                matrix_html += f"<td style='padding: 10px;'>{value}</td>"
            matrix_html += "</tr>"
        matrix_html += "</table>"
        
        # Return the combined HTML with added padding
        return format_html(report_html + matrix_html)

    def model_name(self, obj):
        return obj.prediction.model.model_name
    model_name.short_description = 'Model Name'

    def dataset_name(self, obj):
        return obj.prediction.processed_dataset.dataset.training_file.name.split('/')[-1]
    dataset_name.short_description = 'Dataset Name'

class CustomBestModelAdmin(admin.ModelAdmin):
    list_display = ['market_name', 'model_name', 'best_k', 'save_at']
    list_filter = ['market__market_name']
    readonly_fields = ('market_name', 'model_name', 'best_k','save_at','evaluation_details')
    
    
    @admin.display(description='Evaluation Details')
    def evaluation_details(self, obj):
        evaluation = obj.evaluation
        if not evaluation:
            return "-"

        classification_report = evaluation.classification_report
        confusion_matrix = evaluation.confusion_matrix
        
        # Format classification report with added padding
        report_html = "<h4>Classification Report : </h4><table><tr><th style='padding: 9px;'>Class</th><th style='padding: 9px;'>Precision</th><th style='padding: 9px;'>Recall</th><th style='padding: 9px;'>F1-Score</th><th style='padding: 9px;'>Support</th></tr>"
        for class_label, metrics in classification_report.items():
            if class_label == 'accuracy':
                report_html += f"<tr><td style='padding: 9px;'>{class_label}</td><td style='padding: 9px;'>{metrics:.4f}</td><td> </td><td> </td><td> </td></tr>"
            elif isinstance(metrics, dict):
                report_html += f"<tr><td style='padding: 9px;'>{class_label}</td><td style='padding: 9px;'>{metrics['precision']:.4f}</td><td style='padding: 9px;'>{metrics['recall']:.4f}</td><td style='padding: 9px;'>{metrics['f1-score']:.4f}</td><td style='padding: 9px;'>{metrics['support']}</td></tr>"
        report_html += "</table>"
        
        # Format confusion matrix with added padding
        matrix_html = "<h4>Confusion Matrix :</h4><table>"
        for row in confusion_matrix:
            matrix_html += "<tr>"
            for value in row:
                matrix_html += f"<td style='padding: 10px;'>{value}</td>"
            matrix_html += "</tr>"
        matrix_html += "</table>"
        
        # Return the combined HTML with added padding
        return format_html(report_html + matrix_html)
    def market_name(self, obj):
        return obj.market.market_name
    market_name.short_description = 'Market Name'

    def model_name(self, obj):
        return obj.model.model_name
    model_name.short_description = 'Best Model Name'

class CustomDatasetPredictionAdmin(admin.ModelAdmin):
    list_display = ['market_name', 'predicting_file_name', 'date', 'uploaded_at']

    def market_name(self, obj):
        return obj.market.market_name
    market_name.short_description = 'Market Name'

    def predicting_file_name(self, obj):
        return obj.predicting_file.name.split('/')[-1]
    predicting_file_name.short_description = 'Predicting File Name'

class CustomResultsClientAdmin(admin.ModelAdmin):
    list_display = ['market_name', 'prediction_file_name', 'upload_at']

    def market_name(self, obj):
        return obj.market.market_name
    market_name.short_description = 'Market Name'

    def prediction_file_name(self, obj):
        return obj.dataset_prediction.predicting_file.name.split('/')[-1]
    prediction_file_name.short_description = 'Prediction File Name'

admin.site.register(CustomUser, CustomUserAdmin)
admin.site.register(Market, CustomMarketAdmin)
admin.site.register(Dataset, CustomDatasetAdmin)
admin.site.register(ProcessedDataset, CustomProcessedDatasetAdmin)
admin.site.register(MLModel, CustomMLModelAdmin)
admin.site.register(ModelParameter, CustomModelParameterAdmin)
admin.site.register(Prediction, CustomPredictionAdmin)
admin.site.register(Evaluation, CustomEvaluationAdmin)
admin.site.register(BestModel, CustomBestModelAdmin)
admin.site.register(Dataset_Prediction, CustomDatasetPredictionAdmin)
admin.site.register(Results_Client, CustomResultsClientAdmin)