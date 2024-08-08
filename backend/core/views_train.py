from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponseRedirect,HttpResponseBadRequest
from .models import Market, Dataset, MLModel,ModelParameter, BestModel, Evaluation
from django.urls import reverse
from django.views.decorators.http import require_POST
from .forms import MarketForm, DatasetForm
from .views import *
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import mpld3
from mpld3 import plugins
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
from datetime import datetime
from django.conf import settings
import os
import json
from django.http import StreamingHttpResponse
from time import sleep  # For demonstration purposes

def manage_datasets(request):
    markets = Market.objects.all()
    selected_market = None
    datasets = None

    if request.method == 'POST':
        if 'market' in request.POST:
            # Handle market selection
            market_name = request.POST.get('market')
            selected_market = get_object_or_404(Market, name=market_name)
            datasets = Dataset.objects.filter(market=selected_market).order_by('date')
        elif 'training_file' in request.FILES:
            # Handle dataset upload
            market_id = request.POST.get('market_id')
            market = get_object_or_404(Market, pk=market_id)
            form = DatasetForm(request.POST, request.FILES)
            if form.is_valid():
                dataset = form.save(commit=False)
                dataset.market = market
                dataset.save()
                return redirect('dataset-list')
        elif 'delete_dataset_id' in request.POST:
            # Handle dataset deletion
            dataset_id = request.POST.get('delete_dataset_id')
            dataset = get_object_or_404(Dataset, pk=dataset_id)
            dataset.delete()
            return redirect('dataset-list')
        else:
            # Handle dataset editing
            dataset_id = request.POST.get('dataset_id')
            dataset = get_object_or_404(Dataset, pk=dataset_id)
            date = request.POST.get('date')
            if date:
                dataset.date = date
                dataset.save()
                return redirect('dataset-list')

    form = DatasetForm()
    return render(request, 'Dataset/datasets.html', {
        'markets': markets,
        'selected_market': selected_market,
        'datasets': datasets,
        'form': form
    })
# def main(request):
#     markets = Market.objects.all()
#     selected_market = None
#     datasets = None
#     if request.method == 'POST':
#         market_name = request.POST.get('market')
#         selected_market = get_object_or_404(Market, name=market_name)
#         datasets = Dataset.objects.filter(market=selected_market)
#     return render(request, 'main.html', 
#                 {'markets': markets, 'selected_market': selected_market, 'datasets': datasets})
# def model_recommendations(request, market_name):
#     market = get_object_or_404(Market, name=market_name)
#     models = MLModel.objects.filter(market=market)
#     model_names = [model.name for model in models]
#     return JsonResponse(model_names, safe=False)

# def upload_dataset(request):
#     if request.method == 'POST':
#         market_id = request.POST.get('market')
#         training_file = request.FILES.get('training_file')
#         date = request.POST.get('date')
        
#         if market_id and training_file and date:
#             market = get_object_or_404(Market, pk=market_id)
#             Dataset.objects.create(market=market, training_file=training_file, date=date)
#             return redirect('main')
#         else:
#             return render(request, 'main.html', {'error': 'All fields are required'})
#     return redirect('main')
    
def train(request):
    markets = Market.objects.all()
    selected_market = None
    data_sets = None
    models = MLModel.objects.all()
    print("Model", models)
    if request.method == 'POST':
        market_name = request.POST.get('market')
        selected_market = get_object_or_404(Market, name=market_name)
        data_sets = Dataset.objects.filter(market=selected_market).order_by("date")
    return render(request, 'Train/train.html', 
            {'markets': markets,
            'selected_market': selected_market,
            "data_sets":data_sets,
            "models": models})
def generate_histogram_html(market_name, k):
    combined_label = combine_label(market_name, k)
    data = combined_label.to_numpy()
    # Define the bins to include -1, 0, 1
    bins = np.arange(-1.5, 2.5, 1)
    
    fig, ax = plt.subplots()
    
    # Plot the histogram with the correct bins and edge color
    counts, bins, patches = ax.hist(data, bins=bins, edgecolor='r', rwidth=0.8)
    
    ax.set_title('Histogram of Labels')
    ax.set_xlabel("Label: 1 = up, 0 = Unchanged, -1 = down")
    ax.set_ylabel("Frequencies")
    ax.set_xticks([-1, 0, 1])
    
    histogram_html = mpld3.fig_to_html(fig)
    plt.close(fig)
    return histogram_html

def find_best_k(request, market_name):
    if request.method == 'POST':
        try:
            k_min = int(request.POST.get('k-min', 1))
            k_max = int(request.POST.get('k-max', 100))
            if k_min < 1 or k_max < k_min:
                return JsonResponse({'error': 'Invalid k range.'}, status=400)
            best_k_value = find_best_k_logic(market_name, k_min, k_max)
            histogram_html = generate_histogram_html(market_name, best_k_value)
            
            return JsonResponse({'k': best_k_value, 'histogram_html': histogram_html})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return HttpResponseBadRequest('Invalid request method.')

def get_datasets(request, market_name):
    if request.method == 'GET':
        try:
            market = get_object_or_404(Market, name=market_name)
            datasets = Dataset.objects.filter(market=market).order_by('date')
            dataset_list = []
            for dataset in datasets:
                print(dataset.date)
                dataset_list.append({
                    'id': dataset.id,
                    'date': dataset.date.strftime('%Y-%m-%d'),
                    'name': dataset.training_file.name
                })
            return JsonResponse({'datasets': dataset_list})
        except Market.DoesNotExist:
            return JsonResponse({'error': 'Market not found'}, status=404)
    return JsonResponse({'error': 'Invalid request method'}, status=400)

@require_POST
def preprocess_train(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)

    # Check if the dataset has already been processed
    processed_dataset = ProcessedDataset.objects.filter(dataset=dataset).first()
    
    if processed_dataset and processed_dataset.normalized_file and processed_dataset.normalized_file.path:
        return JsonResponse({'message': 'This dataset has already been preprocessed.'}, status=200)

    # Load the dataset
    df = read_dataset(dataset.training_file.path)
    if df is not None:
        # Clean and normalize the dataframe
        df_cleaned = clean_dataframe(df)
        df_normalized = normalize_dataframe(df_cleaned)

        # Convert dataframes to CSV
        cleaned_csv = df_cleaned.to_csv(index=False)
        normalized_csv = df_normalized.to_csv(index=False)

        # Create a new ProcessedDataset object if not exists
        if not processed_dataset:
            processed_dataset = ProcessedDataset(dataset=dataset)

        # Save cleaned and normalized files
        processed_dataset.cleaned_file.save(f"{dataset.id}_cleaned.csv", ContentFile(cleaned_csv))
        processed_dataset.normalized_file.save(f"{dataset.id}_normalized.csv", ContentFile(normalized_csv))
        processed_dataset.save()

        return JsonResponse({'message': 'Preprocessing completed successfully.'}, status=201)
    else:
        return JsonResponse({'message': 'Failed to read dataset'}, status=400)

@require_POST
def incremental_train_model(request, market_name, model_name):
    try:
        market = get_object_or_404(Market, name=market_name)
        datasets = Dataset.objects.filter(market=market).order_by('date')
        k = int(request.POST.get('k'))
        print("K =", k)

        models = {
            "KNN": KNeighborsClassifier(n_neighbors=7),
            "Ridge Regression": RidgeClassifier(alpha=1.0),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(n_estimators=50)
        }

        model = models.get(model_name)
        if model is None:
            return JsonResponse({'error': 'Model not found'}, status=400)

        results = []
        print("Number of datasets", len(datasets))

        for i in range(0, len(datasets)):
            train_data = datasets[:i]
            print("train_data", train_data)
            test_data = datasets[i:i+1]
            print("test data", test_data)

            X_train_list = []
            Y_train_list = []

            for dataset in train_data:
                print("dataset:", dataset)
                processed_dataset = get_object_or_404(ProcessedDataset, dataset=dataset)
                print("processed data:", processed_dataset)
                normalized_file_path = processed_dataset.normalized_file.path
                print("normalized path:", normalized_file_path)
                df_normalized = pd.read_csv(normalized_file_path)
                X_train = feature_for_k_events(df_normalized, k)
                cleaned_file_path = processed_dataset.cleaned_file.path
                df_cleaned = pd.read_csv(cleaned_file_path)
                mid_price = get_mid_price(df_cleaned)
                Y_train = label_k_event(mid_price, k)

                print("X_train.shape", X_train.shape)
                print("Y_train.shape", Y_train.shape)

                # Adding detailed content check
                print(f"X_train head: {X_train.head()}")
                print(f"Y_train head: {Y_train.head()}")

                if X_train.shape[0] != Y_train.shape[0]:
                    print(f"Shape mismatch for dataset {dataset}: X_train.shape = {X_train.shape}, Y_train.shape = {Y_train.shape}")
                    continue  # Skip this dataset if there's a mismatch

                if len(X_train) == 0 or len(Y_train) == 0:
                    print("Empty training data. Skipping this dataset.")
                    continue

                X_train_list.append(X_train)
                Y_train_list.append(Y_train)

            if not X_train_list or not Y_train_list:
                print("No training data available. Skipping iteration.")
                continue

            X_train = np.vstack(X_train_list)
            Y_train = np.vstack(Y_train_list).ravel()  # Ensuring Y_train is a flat array

            print("Final X_train.shape:", X_train.shape)
            print("Final Y_train.shape:", Y_train.shape)

            try:
                
                model.fit(X_train, Y_train)
            except Exception as fit_err:
                print("Error during model fitting:", fit_err)
                continue

            test_dataset = test_data[0]
            processed_test_dataset = get_object_or_404(ProcessedDataset, dataset=test_dataset)
            test_normalized_file_path = processed_test_dataset.normalized_file.path
            df_test_normalized = pd.read_csv(test_normalized_file_path)
            X_test = feature_for_k_events(df_test_normalized, k)

            cleaned_file_path_test = processed_test_dataset.cleaned_file.path
            df_cleaned_test = pd.read_csv(cleaned_file_path_test)
            mid_price_test = get_mid_price(df_cleaned_test)
            Y_test = label_k_event(mid_price_test, k).values.ravel()

            try:
                score = model.score(X_test, Y_test)
            except Exception as score_err:
                print("Error during model scoring:", score_err)
                continue

            # Save prediction
            model_object = get_object_or_404(MLModel, name=model_name) 
            prediction = Prediction.objects.create(
                model=model_object,
                processed_dataset=processed_test_dataset,
                predictions = json.dumps(model.predict(X_test).tolist()),
            )

            # Evaluate the prediction
            evaluation = Evaluation.objects.create(
                prediction=prediction,
                accuracy=score,
                classification_report=classification_report(Y_test, model.predict(X_test), output_dict=True),
                confusion_matrix=confusion_matrix(Y_test, model.predict(X_test)).tolist()
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{market_name}_{model_name}_{timestamp}_model_{i}.joblib"
            model_file_path = os.path.join(settings.MEDIA_ROOT, "trained_models", model_filename)
            joblib.dump(model, model_file_path)

            ml_model_instance, created = MLModel.objects.get_or_create(name=model_name)
            with open(model_file_path, 'rb') as model_file:
                ModelParameter.objects.create(
                    model=ml_model_instance,
                    processed_dataset=processed_test_dataset,
                    parameters=model.get_params(),
                    model_file=ContentFile(model_file.read(), name=model_filename),
                    k = k
                )

            # Append current training status
            results.append({
                'iteration': i,
                'model': model_name,
                'score': score,
                'dataset_date': test_dataset.date,
                'evaluation_id': evaluation.id,
                'classification_report': evaluation.classification_report,
                'confusion_matrix': evaluation.confusion_matrix,
                'accuracy': evaluation.accuracy
            })
            print("evaluation_id: ",evaluation.id)
            print("confusion matrix", evaluation.classification_report)
            print("classification report", evaluation.classification_report)

        return JsonResponse({'results': results, 'complete': True})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)

@require_POST
def save_best_model(request):    

    try:
        market_name = request.POST.get('market')
        model_name = request.POST.get('model')
        evaluation_id = request.POST.get('evaluation')
        best_k = request.POST.get('best_k')

        market = get_object_or_404(Market, name=market_name)
        model = get_object_or_404(MLModel, name=model_name)
        evaluation = get_object_or_404(Evaluation, id=evaluation_id)

        best_model = BestModel(market=market, model=model, evaluation=evaluation, best_k= best_k)
        best_model.save()

        return JsonResponse({'status': 'Best model saved successfully'})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)



