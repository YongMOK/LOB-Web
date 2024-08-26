from django.shortcuts import render, get_object_or_404
from .models import Market, Dataset_Prediction, BestModel, Results_Client
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from datetime import datetime, timedelta
import pandas as pd
from django.core.files.base import ContentFile
from .views import clean_dataframe, normalize_dataframe, feature_for_k_events
from .helpers import generate_order_book
from .views_prediction import predictions_model, save_results_to_client, create_result_dataframe
import json
import pytz
paris_tz = pytz.timezone('Europe/Paris')


@login_required
def client(request):
    markets = Market.objects.all()
    selected_market = None
    results = None

    if request.method == 'POST':
        market_name = request.POST.get('market')
        selected_market = get_object_or_404(Market, market_name=market_name)
        #results = generate_and_predict_real_time(selected_market)
        try:
            results = Results_Client.objects.filter(market=selected_market).latest("upload_at")
        except Results_Client.DoesNotExist:
            results = None

    return render(request, 'prediction/client_result.html', {
        'markets': markets,
        'selected_market': selected_market,
        'results': results,
    })


def generate_and_predict_real_time(market):
    today_date = timezone.localdate()
    print(today_date)
    time_now = timezone.now()

    try:
        predictions_dataset = Dataset_Prediction.objects.filter(market=market, date=today_date).latest("uploaded_at")
        #print("predictions_dataset",predictions_dataset)
        if not predictions_dataset:
            raise Dataset_Prediction.DoesNotExist("No prediction dataset available for today.")

        results = Results_Client.objects.filter(market=market, dataset_prediction = predictions_dataset).latest("upload_at")
        print("Result",results)
        # results_last_time = timezone.make_aware(datetime.fromisoformat(results.result["timestamp"][-2]))
        # Assuming results.result["timestamp"] is a list of ISO formatted date strings
        result_data = json.loads(results.result)
        last_timestamp_ms = result_data[-2]['timestamp']
        # Convert milliseconds to seconds (as required by datetime.fromtimestamp)
        last_datetime = datetime.fromtimestamp(last_timestamp_ms / 1000.0, paris_tz)
        print("last_datetime",last_datetime)
        #results_last_time = timezone.make_aware(last_datetime)
        results_last_time = last_datetime

        print("time_now",time_now)

        if (time_now - results_last_time) <= timedelta(seconds=10):
            return results.result
        
        df_existing = pd.read_csv(predictions_dataset.predicting_file.path)
        last_timestamp = results_last_time

    except Dataset_Prediction.DoesNotExist:
        df_existing = pd.DataFrame()
        last_timestamp = datetime.combine(today_date, market.start_time)
    print("results last time",results_last_time)   
    print("time_now",time_now)
    missing_data = generate_order_book(last_timestamp, time_now)
    df_missing_data = pd.DataFrame(missing_data)
    df_combined = pd.concat([df_existing, df_missing_data], ignore_index=True)
    print("data_combined",df_combined)
    print("saving data.........")
    save_updated_dataset(predictions_dataset, df_combined, market)
    print("Finished saving data........")

    print("preparing data........")
    df_cleaned = clean_dataframe(df_combined)
    df_normalized = normalize_dataframe(df_cleaned)
    best_model = BestModel.objects.filter(market=market).latest('save_at')
    k = best_model.best_k
    X_test = feature_for_k_events(df_normalized, k)
    print("finised preparing data.......")
    predictions = predictions_model(best_model.model.name, X_test, k)

    result_df = create_result_dataframe(df_combined["Timestamp"], predictions, k)
    save_results_to_client(market, predictions_dataset, result_df)
    print(result_df.to_json(orient='records'))
    return result_df.to_json(orient='records')

def save_updated_dataset(predictions_dataset, df_combined, market):
    file_name = f"{market.name}_{timezone.now().date()}.csv"

    if predictions_dataset:
        predictions_dataset.predicting_file.delete(save=False)  # Optionally delete the old file

    csv_content = df_combined.to_csv(index=False)
    predictions_dataset.predicting_file.save(file_name, ContentFile(csv_content.encode('utf-8')))

