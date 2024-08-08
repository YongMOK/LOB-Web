
from rest_framework import serializers
from .models import Market, Dataset, ProcessedDataset, MLModel, ModelParameter, Prediction, Evaluation

class MarketSerializer(serializers.ModelSerializer):
    class Meta:
        model = Market
        fields = '__all__'

class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = '__all__'

class ProcessedDatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProcessedDataset
        fields = '__all__'

class MLModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLModel
        fields = '__all__'

class ModelParameterSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelParameter
        fields = '__all__'

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = '__all__'

class EvaluationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Evaluation
        fields = '__all__'
