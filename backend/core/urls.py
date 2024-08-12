from django.urls import path
from django.contrib.auth.decorators import login_required, user_passes_test
from rest_framework.routers import DefaultRouter
from .views import (
    MarketViewSet, DatasetViewSet, ProcessedDatasetViewSet,
    MLModelViewSet, ModelParameterViewSet, PredictionViewSet, EvaluationViewSet
)
from . import views_train, views_prediction,views_log_in_out,views_home, views_client
from django.contrib.auth.views import (
    LogoutView, 
    PasswordResetView, 
    PasswordResetDoneView, 
    PasswordResetConfirmView,
    PasswordResetCompleteView
)

router = DefaultRouter()
router.register(r'markets', MarketViewSet, basename='market')
router.register(r'datasets', DatasetViewSet, basename='dataset')
router.register(r'processed_datasets', ProcessedDatasetViewSet, basename='processed_dataset')
router.register(r'models', MLModelViewSet, basename='model')
router.register(r'model_parameters', ModelParameterViewSet, basename='model_parameter')
router.register(r'predictions', PredictionViewSet, basename='prediction')
router.register(r'evaluations', EvaluationViewSet, basename='evaluation')

def admin_required(view_func):
    return user_passes_test(lambda u: u.is_superuser)(view_func)

urlpatterns = [
    path('', views_log_in_out.login_view, name='login'),
    path('home/', login_required(views_home.home), name='home'),
    path('logout/', views_log_in_out.logout_view, name='logout'),
    path('register/',views_log_in_out.register, name='register'),
    path('password_reset/',PasswordResetView.as_view(template_name='Login_out/password_reset.html'), name='password_reset'),
    path('password_reset_done/', PasswordResetDoneView.as_view(template_name='Login_out/password_reset_done.html'), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', PasswordResetConfirmView.as_view(template_name='Login_out/password_reset_confirm.html'), name='password_reset_confirm'),
    path('reset/complete/', PasswordResetCompleteView.as_view(template_name='Login_out/password_reset_complete.html'), name='password_reset_complete'),
    # path('main/',login_required(views_train.main), name='main'),
    # path('api/', include(router.urls)),
    path('markets/', admin_required(MarketViewSet.as_view({'get': 'list'})), name='market-list'),
    path('markets/add/', admin_required(MarketViewSet.as_view({'post': 'add_market'})), name='add_market'),
    path('markets/<int:pk>/edit/', admin_required(MarketViewSet.as_view({'post': 'edit_market'})), name='market-edit'),
    path('markets/<int:pk>/delete/', admin_required(MarketViewSet.as_view({'post': 'delete_market'})), name='market-delete'),
    
    path('datasets/', admin_required(views_train.manage_datasets), name='dataset-list'),
    # path('models/', MLModelViewSet.as_view({'get': 'list'}), name='mlmodel-list'),
    # path('models/add/', MLModelViewSet.as_view({'post': 'add_model'}), name='mlmodel-add'),
    # path('models/<int:pk>/edit/', MLModelViewSet.as_view({'post': 'edit_model'}), name='mlmodel-edit'),
    # path('models/<int:pk>/delete/', MLModelViewSet.as_view({'post': 'delete_model'}), name='mlmodel-delete'),
    # path('model_parameters/train/', ModelParameterViewSet.as_view({'post': 'train'}),name='modelparameter-train'),
    # path('model_recommendations/<str:market_name>/', (views_train.model_recommendations), name='model-recommendations'),
    # path('predictions/predict/', PredictionViewSet.as_view({'post': 'predict'}), name='prediction-predict'),
    # path('upload_dataset/', login_required(views_train.upload_dataset), name='upload_dataset'),
    path('train/', admin_required(views_train.train), name='train'),
    path('find_best_k/<str:market_name>/', admin_required(views_train.find_best_k), name='find_best_k'),
    path('get_datasets/<str:market_name>/', admin_required(views_train.get_datasets), name='get_datasets'),
    path('preprocess_train/<int:dataset_id>/', admin_required(views_train.preprocess_train), name='preprocess_train'),
    path('incremental_train_model/<str:market_name>/<str:model_name>/', admin_required(views_train.incremental_train_model), name='incremental_train_model'),
    path('save_best_model/', admin_required(views_train.save_best_model), name='save_best_model'),
    
    path('predictions/', admin_required(views_prediction.prediction), name='predictions'),
    path('upload_predictions/', admin_required(views_prediction.upload_predictions), name='upload_predictions'),
    
    
    
    path('client_result/', login_required(views_client.client), name='client_result'),  
]

