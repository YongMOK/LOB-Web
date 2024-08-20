from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
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

admin.site.register(CustomUser, CustomUserAdmin)
admin.site.register(Market)
admin.site.register(Dataset)
admin.site.register(ProcessedDataset)
admin.site.register(MLModel)
admin.site.register(ModelParameter)
admin.site.register(Evaluation)
admin.site.register(Prediction)
admin.site.register(BestModel)
admin.site.register(Dataset_Prediction)
admin.site.register(Results_Client)