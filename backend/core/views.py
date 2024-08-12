from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import action
from .models import Market, Dataset, ProcessedDataset, MLModel, ModelParameter, Prediction, Evaluation
from .serializers import (
    MarketSerializer, DatasetSerializer, ProcessedDatasetSerializer,
    MLModelSerializer, ModelParameterSerializer, PredictionSerializer, EvaluationSerializer
)
from django.shortcuts import get_object_or_404
from .forms import MarketForm, DatasetForm, MLModelForm, UploadFileForm
from django.contrib.auth.decorators import login_required
from django.contrib.admin.views.decorators import staff_member_required
from django.shortcuts import render, redirect
from django.urls import reverse
from django.conf import settings
from django.core.files.base import ContentFile
from django.http import HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.http import JsonResponse
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib
from django.db import IntegrityError

# ML functions
def read_dataset(file_path):
    try:
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        data = pd.read_csv(full_path)
        return data
    except Exception as e:
        print(f"Error reading dataset from {file_path}: {e}")
        return None

def clean_dataframe(df):
    df = df.drop(columns=df.columns[0], axis=1)
    df_cleaned = df.dropna()
    return df_cleaned

def normalize_dataframe(df):
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_normalized

def feature_for_k_events(data, k):
    num_new_rows = data.shape[0] - k + 1
    num_new_cols = data.shape[1] * k
    df_new = np.zeros((num_new_rows, num_new_cols))

    for i in range(num_new_rows):
        df_new[i] = data.iloc[i:i+k, :].values.flatten()
    return pd.DataFrame(df_new)

def label_k_event(mid_price, k):
    length = mid_price.shape[0]
    label = np.zeros(length-k+1)
    for i in range(length-k):
        mean_k = mid_price[i:i+k].mean()
        percentage_change = (mean_k - mid_price[i])/mid_price[i]
        if percentage_change >= 0.0002:
            label[i] = 1
        elif -0.000199 <= percentage_change < 0.000199:
            label[i] = 0
        elif percentage_change < -0.0002:
            label[i] = -1
    label[length-k] = -1
    return pd.DataFrame(label)

def get_mid_price(data):
    df = (data.iloc[:, 0] + data.iloc[:, 2])/2
    return df


def label_proportion(df):
    count_1 = (df == 1).sum()
    count_0 = (df == 0).sum()
    count_m1 = (df == -1).sum()
    count = df.shape[0]
    p = np.array([count_1, count_0, count_m1])
    return p / count

def combine_label(market_name, k):
    market = get_object_or_404(Market, name=market_name)
    datasets = Dataset.objects.filter(market=market)
    combined_label = pd.DataFrame()
    for dataset in datasets:
        
        data = read_dataset(dataset.training_file.path)
        data = clean_dataframe(data)
        mid_price = get_mid_price(data)
        label = label_k_event(mid_price, k)
        print(label)
        combined_label = pd.concat([combined_label, label], axis=0)
    return combined_label


def find_best_k_logic(market_name, k_min, k_max):
    list_k_proportion = []
    if k_min == k_max :
        best_k = k_min
        return best_k
    for k in range(k_min, k_max + 1):
        label = combine_label(market_name, k)
        p = label_proportion(label)
        probab_k = p[0] * p[1] * p[2]
        list_k_proportion.append(probab_k)
    best_k = list_k_proportion.index(max(list_k_proportion)) + k_min
    return best_k
class MarketViewSet(viewsets.ModelViewSet):
    queryset = Market.objects.all()
    serializer_class = MarketSerializer
    
    def list(self, request, *args, **kwargs):
        if request.accepted_renderer.format == 'html' or 'text/html' in request.META.get('HTTP_ACCEPT'):
            markets = self.get_queryset()
            return render(request, 'Market/market.html', {'markets': markets})
        else:
            return super().list(request, *args, **kwargs)
    @action(detail=False, methods=['post'], url_path='add')
    def add_market(self, request):
        name = request.POST.get('name')
        start_time = request.POST.get('start_time')
        end_time = request.POST.get('end_time')
        if name:
            Market.objects.create(name=name, end_time=end_time, start_time=start_time)
            return HttpResponseRedirect(reverse('market-list'))
        return render(request, 'Market/market.html',{'error':'Market name is required'})
    @action(detail=True, methods=['post'], url_path='edit')	
    def add_market(self, request):
        name = request.POST.get('name')
        start_time = request.POST.get('start_time')
        end_time = request.POST.get('end_time')

        if name:
            try:
                Market.objects.create(name=name, start_time=start_time, end_time=end_time)
                return HttpResponseRedirect(reverse('market-list'))
            except IntegrityError:
                markets = Market.objects.all()
                error_message = f'Market with name "{name}" already exists.'
                return render(request, 'Market/market.html', {
                    'markets': markets,
                    'error': error_message,
                })
        else:
            return render(request, 'Market/market.html', {'error': 'Market name is required'})

    @action(detail=True, methods=['post'], url_path='delete')
    def delete_market(self, request,pk):
        market = get_object_or_404(Market, pk=pk)
        market.delete()
        return HttpResponseRedirect(reverse('market-list'))
        
    def retrieve(self, request, pk=None):
        queryset = self.get_queryset()
        market = get_object_or_404(queryset, pk=pk)
        serializer = MarketSerializer(market)
        return Response(serializer.data)

class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    #permission_classes = [IsAuthenticated]

    def list(self, request, *args, **kwargs):
        if request.accepted_renderer.format == 'html' or 'text/html' in request.META.get('HTTP_ACCEPT'):
            datasets = self.get_queryset()
            return render(request, 'dataset.html', {'datasets': datasets})
        else:
            return super().list(request, *args, **kwargs)

    @action(detail=False, methods=['post'], url_path='upload')
    def upload_dataset(self, request):
        market_id = request.POST.get('market')
        training_file = request.FILES.get('training_file')
        date = request.POST.get('date')

        if market_id and training_file and date:
            market = get_object_or_404(Market, pk=market_id)
            Dataset.objects.create(market=market, training_file=training_file, date=date)
            return HttpResponseRedirect(reverse('dataset-list'))
        return render(request, 'dataset.html', {'error': 'All fields are required'})

    @action(detail=True, methods=['post'], url_path='edit')
    def edit_dataset(self, request, pk=None):
        dataset = get_object_or_404(Dataset, pk=pk)
        market_id = request.POST.get('market')
        date = request.POST.get('date')

        if market_id and date:
            market = get_object_or_404(Market, pk=market_id)
            dataset.market = market
            dataset.date = date
            dataset.save()
            return HttpResponseRedirect(reverse('dataset-list'))
        return render(request, 'dataset.html', {'error': 'All fields are required'})

    @action(detail=True, methods=['post'], url_path='delete')
    def delete_dataset(self, request, pk=None):
        dataset = get_object_or_404(Dataset, pk=pk)
        dataset.delete()
        return HttpResponseRedirect(reverse('dataset-list'))

class ProcessedDatasetViewSet(viewsets.ModelViewSet):
    queryset = ProcessedDataset.objects.all()
    serializer_class = ProcessedDatasetSerializer
    #permission_classes = [IsAuthenticated]

    def list(self, request):
        queryset = self.get_queryset()
        serializer = ProcessedDatasetSerializer(queryset, many=True)
        return Response(serializer.data)

    def create(self, request):
        serializer = ProcessedDatasetSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def retrieve(self, request, pk=None):
        queryset = self.get_queryset()
        processed_dataset = get_object_or_404(queryset, pk=pk)
        serializer = ProcessedDatasetSerializer(processed_dataset)
        return Response(serializer.data)

    def update(self, request, pk=None):
        processed_dataset = get_object_or_404(ProcessedDataset, pk=pk)
        serializer = ProcessedDatasetSerializer(processed_dataset, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None):
        processed_dataset = get_object_or_404(ProcessedDataset, pk=pk)
        processed_dataset.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    @method_decorator(csrf_exempt)
    @action(detail=True, methods=['post'], url_path='preprocess')
    def preprocess(self, request, pk=None):
        dataset = get_object_or_404(Dataset, id=pk)
        df = read_dataset(dataset.training_file.path)

        if ProcessedDataset.objects.filter(dataset=dataset).exists():
            return Response({'message': 'This dataset has already been preprocessed.'}, status=status.HTTP_200_OK)

        if df is not None:
            df_cleaned = clean_dataframe(df)
            df_normalized = normalize_dataframe(df_cleaned)
            cleaned_csv = df_cleaned.to_csv(index=False)
            normalized_csv = df_normalized.to_csv(index=False)

            processed_dataset = ProcessedDataset(
                dataset=dataset,
            )
            processed_dataset.cleaned_file.save(f"{dataset.id}_cleaned.csv", ContentFile(cleaned_csv))
            processed_dataset.normalized_file.save(f"{dataset.id}_normalized.csv", ContentFile(normalized_csv))
            processed_dataset.save()

            return Response({'message': 'Preprocessing completed successfully.'}, status=status.HTTP_201_CREATED)
        else:
            return Response({'message': 'Failed to read dataset'}, status=status.HTTP_400_BAD_REQUEST)
class MLModelViewSet(viewsets.ModelViewSet):
    queryset = MLModel.objects.all()
    serializer_class = MLModelSerializer
    permission_classes = [IsAuthenticated]

    def list(self, request, *args, **kwargs):
        if request.accepted_renderer.format == 'html' or 'text/html' in request.META.get('HTTP_ACCEPT'):
            queryset = self.get_queryset()
            markets = Market.objects.all()  # Fetch all markets
            return render(request, "model.html", {'models': queryset, 'markets': markets})
        else:
            return super().list(request, *args, **kwargs)
    @action(detail=False, methods=['post'], url_path='add')
    def add_model(self, request):
        name = request.data.get('name')
        market_id = request.data.get('market')
        if name and market_id:
            market = get_object_or_404(Market, id=market_id)
            MLModel.objects.create(name=name, market=market)
            return redirect('mlmodel-list')
        markets = Market.objects.all()
        return render(request, 'model.html', {'error': 'Model name and market are required', 'models': self.get_queryset(), 'markets': markets})

    @action(detail=True, methods=['post'], url_path='edit')	
    def edit_model(self, request, pk):
        model = get_object_or_404(MLModel,pk=pk)
        name = request.POST.get('name')
        if name:
            model.name = name
            model.save()
            return redirect('mlmodel-list')
        return render(request, 'model.html', {'error': 'Model name is required', 'models': self.get_queryset(), 'markets': Market.objects.all()})
    @action(detail=True, methods=['post'], url_path='delete')
    def delete_model(self, request,pk):
        model = get_object_or_404(MLModel, pk=pk)
        model.delete()
        return HttpResponseRedirect(reverse('mlmodel-list'))

    def retrieve(self, request, pk=None):
        queryset = self.get_queryset()
        ml_model = get_object_or_404(queryset, pk=pk)
        serializer = MLModelSerializer(ml_model)
        return Response(serializer.data)
class ModelParameterViewSet(viewsets.ModelViewSet):
    queryset = ModelParameter.objects.all()
    serializer_class = ModelParameterSerializer
    #permission_classes = [IsAuthenticated]

    def list(self, request):
        queryset = self.get_queryset()
        serializer = ModelParameterSerializer(queryset, many=True)
        return Response(serializer.data)

    def create(self, request):
        serializer = ModelParameterSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def retrieve(self, request, pk=None):
        queryset = self.get_queryset()
        model_parameter = get_object_or_404(queryset, pk=pk)
        serializer = ModelParameterSerializer(model_parameter)
        return Response(serializer.data)

    def update(self, request, pk=None):
        model_parameter = get_object_or_404(ModelParameter, pk=pk)
        serializer = ModelParameterSerializer(model_parameter, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None):
        model_parameter = get_object_or_404(ModelParameter, pk=pk)
        model_parameter.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(detail=False, methods=['post'])
    def train(self, request):
        market_name = request.data.get('market_name')
        model_name = request.data.get('model_name') or request.data.get('existing_model')
        dataset_id = request.data.get('dataset_id')
        k = int(request.data.get('k', 5))
        force_retrain = request.data.get('force_retrain', False)

        try:
            # Get the Market
            market = get_object_or_404(Market, name=market_name)

            # Get the Dataset
            dataset = get_object_or_404(Dataset, id=dataset_id, market=market)

            # Read and process the dataset
            df = read_dataset(dataset.training_file.path)
            if not ProcessedDataset.objects.filter(dataset=dataset).exists():
                df_cleaned = clean_dataframe(df)
                df_normalized = normalize_dataframe(df_cleaned)
                cleaned_csv = df_cleaned.to_csv(index=False)
                normalized_csv = df_normalized.to_csv(index=False)
                processed_dataset = ProcessedDataset(
                    dataset=dataset
                )
                processed_dataset.cleaned_file.save(f"{dataset.id}_cleaned.csv", ContentFile(cleaned_csv))
                processed_dataset.normalized_file.save(f"{dataset.id}_normalized.csv", ContentFile(normalized_csv))
                processed_dataset.save()
            else:
                processed_dataset = get_object_or_404(ProcessedDataset, dataset=dataset)
                # Get the cleaned dataset
                cleaned_file_path = processed_dataset.cleaned_file.path
                df_cleaned = pd.read_csv(cleaned_file_path)
                # Get the normalized dataset
                normalized_file_path = processed_dataset.normalized_file.path
                df_normalized = pd.read_csv(normalized_file_path)

            # Get or create the ML model instance
            ml_model_instance, created = MLModel.objects.get_or_create(
                market=market,
                name=model_name
            )
            # Check if the ML model has been trained with this processed dataset
            if ModelParameter.objects.filter(model=ml_model_instance, processed_dataset=processed_dataset).exists():
                if not force_retrain:
                    return Response({"message": f"{ml_model_instance.name} has already been trained with this dataset.", "already_trained": True}, status=status.HTTP_200_OK)
            # Train the model
            result = self.train_and_save_model(market_name, model_name, df_cleaned, df_normalized, k, processed_dataset, ml_model_instance)
            return Response({'result': result}, status=status.HTTP_200_OK)
        
        except Market.DoesNotExist:
            return Response({'message': f"Market '{market_name}' not found"}, status=status.HTTP_404_NOT_FOUND)
        except Dataset.DoesNotExist:
            return Response({'message': f"Dataset with ID '{dataset_id}' not found in market '{market_name}'"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({'message': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    def train_and_save_model(self, market_name, model_name, cleaned_data, normalized_data, k, processed_dataset, ml_model_instance):
        try:
            # Extract features and labels
            X_train = feature_for_k_events(normalized_data, k)
            mid_price = get_mid_price(cleaned_data)
            Y_train = label_k_event(mid_price, k)

            # Select the model
            model_dict = {
                "KNN": KNeighborsClassifier(n_neighbors=7),
                "Ridge Regression": RidgeClassifier(alpha=1.0),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(n_estimators=50),
            }

            model = model_dict.get(model_name)
            if model is None:
                return {"error": "Model not found"}

            # Train the model
            model.fit(X_train, Y_train.values.ravel())

            # Save the model for future use
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{market_name}_{model_name}_{timestamp}_model.joblib"
            model_file_path = os.path.join(settings.MEDIA_ROOT, "trained_models", model_filename)
            joblib.dump(model, model_file_path)

            # Save the model file path to the database
            with open(model_file_path, 'rb') as model_file:
                ModelParameter.objects.create(
                    model=ml_model_instance,
                    processed_dataset=processed_dataset,
                    parameters=model.get_params(),
                    model_file=ContentFile(model_file.read(), name=model_filename)
                )

            return {
                "status": "Model trained and parameters saved successfully",
                "model_filename": model_filename
            }
        except Exception as e:
            return {"error": str(e)}


class PredictionViewSet(viewsets.ModelViewSet):
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer
    #permission_classes = [IsAuthenticated]

    def list(self, request):
        queryset = self.get_queryset()
        serializer = PredictionSerializer(queryset, many=True)
        return Response(serializer.data)

    def create(self, request):
        serializer = PredictionSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def retrieve(self, request, pk=None):
        queryset = self.get_queryset()
        prediction = get_object_or_404(queryset, pk=pk)
        serializer = PredictionSerializer(prediction)
        return Response(serializer.data)

    def update(self, request, pk=None):
        prediction = get_object_or_404(Prediction, pk=pk)
        serializer = PredictionSerializer(prediction, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None):
        prediction = get_object_or_404(Prediction, pk=pk)
        prediction.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(detail=False, methods=['post'])
    def predict(self, request):
        market_name = request.data.get('market_name')
        model_name = request.data.get('model_name')
        dataset_id = request.data.get('dataset_id')
        k = int(request.data.get('k', 5))

        try:
            # Get the Market
            market = get_object_or_404(Market, name=market_name)

            # Get the Datasets
            datasets = Dataset.objects.filter(market=market, id=dataset_id)

            if not datasets.exists():
                return Response({'message': f"No dataset found for id '{dataset_id}' in market '{market_name}'"}, status=status.HTTP_404_NOT_FOUND)

            # Assuming you want to use the last dataset if there are multiple
            dataset = datasets.last()

            # Read the dataset and process the dataset
            df = read_dataset(dataset.training_file.path)

            if not ProcessedDataset.objects.filter(dataset=dataset).exists():
                df_cleaned = clean_dataframe(df)
                df_normalized = normalize_dataframe(df_cleaned)
                cleaned_csv = df_cleaned.to_csv(index=False)
                normalized_csv = df_normalized.to_csv(index=False)
                processed_dataset = ProcessedDataset(
                    dataset=dataset
                )
                processed_dataset.cleaned_file.save(f"{dataset.id}_cleaned.csv", ContentFile(cleaned_csv))
                processed_dataset.normalized_file.save(f"{dataset.id}_normalized.csv", ContentFile(normalized_csv))
                processed_dataset.save()
            else:
                processed_dataset = ProcessedDataset.objects.get(dataset=dataset)
                # Get the cleaned dataset
                cleaned_file_path = processed_dataset.cleaned_file.path
                df_cleaned = pd.read_csv(cleaned_file_path)
                # Get the normalized dataset
                normalized_file_path = processed_dataset.normalized_file.path
                df_normalized = pd.read_csv(normalized_file_path)

            X_test = feature_for_k_events(df_normalized, k)
            mid_price = get_mid_price(df_cleaned)
            Y = label_k_event(mid_price, k)

            result = self.predictions_model(market_name, model_name, X_test)
            
            # Save predictions
            prediction = Prediction.objects.create(
                model=MLModel.objects.get(name=model_name, market=processed_dataset.dataset.market),
                processed_dataset=processed_dataset,
                predictions=json.dumps(result.tolist())
            )
            serializer = PredictionSerializer(prediction)
            # Evaluate model
            evaluation_data = self.evaluate(Y, result, prediction)
            return JsonResponse({"result":{
                    'status': 'Prediction successful',
                    'data': serializer.data,
                    'accuracy':evaluation_data['accuracy'],
                    'classification_report': evaluation_data['report'],
                    'confusion_matrix': evaluation_data['matrix']
                }}, status=status.HTTP_200_OK)
            
        except Exception as e:
            # Log the error
            print(f"Prediction error: {e}")
            return Response({'message': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    def predictions_model(self, market_name, model_name, X_test):
        market = get_object_or_404(Market, name=market_name)
        ml_model = get_object_or_404(MLModel, market=market, name=model_name)

        # Retrieve the latest ModelParameter instance related to this model
        latest_model_parameter = ModelParameter.objects.filter(model=ml_model).order_by('-trained_at').first()

        if not latest_model_parameter:
            return {"error": f"No parameters found for model '{model_name}' in market '{market_name}'."}

        # Load the trained model from the file
        model_file_path = latest_model_parameter.model_file.path
        model = joblib.load(model_file_path)

        # Use the loaded model to make predictions
        Y = model.predict(X_test)
        return Y

    def evaluate(self, Y, Y_predict, prediction):
        # Custom logic to evaluate predictions
        accuracy, report, matrix = self.evaluate_predictions(Y_predict, Y)
        #print(report)
        evaluation = Evaluation.objects.create(
            prediction=prediction,
            accuracy=accuracy,
            classification_report=json.dumps(report),
            confusion_matrix=json.dumps(matrix)
        )
        serializer = EvaluationSerializer(evaluation)
        return {
            "accuracy": accuracy,
            "report": report,
            "matrix": matrix
        }

    def evaluate_predictions(self, predictions, actual_labels):
        # Convert predictions and actual labels to lists if they are not already
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()
        if isinstance(actual_labels, np.ndarray):
            actual_labels = actual_labels.tolist()

        accuracy = accuracy_score(actual_labels, predictions)
        report = classification_report(actual_labels, predictions, output_dict=True)
        matrix = confusion_matrix(actual_labels, predictions).tolist()  # Convert confusion matrix to list
        
        return accuracy, report, matrix

class EvaluationViewSet(viewsets.ModelViewSet):
    queryset = Evaluation.objects.all()
    serializer_class = EvaluationSerializer
    #permission_classes = [IsAuthenticated]