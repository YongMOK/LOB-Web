from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import action
from .models import Market, Dataset
from .serializers import (
    MarketSerializer, 
)
from django.shortcuts import get_object_or_404
from django.shortcuts import render
from django.urls import reverse
from django.conf import settings
from django.http import HttpResponseRedirect
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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
    market = get_object_or_404(Market, market_name=market_name)
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
            Market.objects.create(market_name=name, closing_time=end_time, opening_time=start_time)
            return HttpResponseRedirect(reverse('market-list'))
        return render(request, 'Market/market.html',{'error':'Market name is required'})
    @action(detail=True, methods=['post'], url_path='edit')	
    def add_market(self, request):
        name = request.POST.get('name')
        start_time = request.POST.get('start_time')
        end_time = request.POST.get('end_time')

        if name:
            try:
                Market.objects.create(market_name=name, opening_time=start_time, closing_time=end_time)
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
