from django.shortcuts import render, get_object_or_404
from .models import Market, Dataset, MLModel, ModelParameter, BestModel, ResultsClient, DatasetPrediction
import numpy as np
import pandas as pd
import joblib
from django.http import JsonResponse
from django.core.exceptions import ObjectDoesNotExist
from .views import *
from dateutil import parser

def prediction(request):
    markets = Market.objects.all()
    selected_market = None
    datasets = None
    best_model = None

    if request.method == 'POST':
        market_name = request.POST.get('market')
        selected_market = get_object_or_404(Market, market_name=market_name)
        datasets = Dataset.objects.filter(market=selected_market).order_by("date")
        try:
            best_model = BestModel.objects.filter(market=selected_market).latest('save_at')
        except ObjectDoesNotExist:
            best_model = None

    return render(request, 'prediction/prediction.html', {
        'markets': markets,
        'selected_market': selected_market,
        'datasets': datasets,
        'best_model': best_model
    })

def upload_predictions(request):
    if request.method == 'POST':
        market_name = request.POST.get('market_name')
        best_model = request.POST.get('best_model')
        k = int(request.POST.get('best_k'))
        prediction_file = request.FILES.get('predicting_file')
        if prediction_file and market_name:
            market = get_object_or_404(Market, market_name = market_name)
            print("Uploading prediction",prediction_file)
            try:
                print('Error..........')
                if prediction_file.name.endswith('.csv'):
                    data = pd.read_csv(prediction_file)
                else:
                    data = read_txt(prediction_file)

                # Preprocess dataset
                dataset_prediction = DatasetPrediction.objects.create(market=market, predicting_file=prediction_file)
                print(data)
                df_cleaned = clean_dataframe(data)
                df_normalized = normalize_dataframe(df_cleaned)
                

                X_test = feature_for_k_events(df_normalized, k)
                print('X_test', X_test)
                result = predictions_model(best_model, X_test, k)
                result_df = create_result_dataframe(data.iloc[:, 0], result, k)
                save_results_to_client(market, dataset_prediction,result_df)
                

                return JsonResponse({
                    "message": "Model prediction successful and saved for client usage",
                    "result": result_df.to_json(orient='records')  # Ensure the JSON format is correct
                })
            except Exception as e:
                return JsonResponse({"error": str(e)}, status=500)

        return JsonResponse({"error": "Invalid input provided."}, status=400)

def predictions_model(model_name, X_test, k):
    try:
        ml_model = get_object_or_404(MLModel, model_name=model_name)
        latest_model_parameter = ModelParameter.objects.filter(model=ml_model, k=k).latest('trained_at')

        if not latest_model_parameter:
            raise ObjectDoesNotExist(f"No parameters found for model '{model_name}'")

        model_file_path = latest_model_parameter.model_file.path
        model = joblib.load(model_file_path)

        Y = model.predict(X_test)
        return Y
    except ObjectDoesNotExist as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}

def read_txt(uploaded_file):
    try:
        data = np.loadtxt(uploaded_file)
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        return None

def create_result_dataframe(timestamps, predictions, k):
    timestamps = [parser.parse(str(ts)) for ts in timestamps.tolist()]
    last_timestamp = timestamps[-1]
    additional_timestamp = last_timestamp + pd.Timedelta(seconds=1.5 * k)

    new_timestamp = timestamps + [additional_timestamp]
    predictions = predictions.tolist()
    new_prediction = [0] * k + predictions

    predictions_accumulator = np.add.accumulate(new_prediction)

    result_df = pd.DataFrame({
        'timestamp': new_timestamp,
        'prediction': predictions_accumulator.tolist()
    })
    return result_df


def save_results_to_client(market, dataset_prediction, results):
    ResultsClient.objects.create(market=market, dataset_prediction = dataset_prediction, result=results.to_json(orient='records'))
