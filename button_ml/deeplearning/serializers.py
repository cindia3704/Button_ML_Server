from rest_framework import serializers, fields
from .models import Cloth_S


class Cloth_SSerializer(serializers.ModelSerializer):
    photo = serializers.ImageField(
        use_url=True, max_length=None, required=False)

    class Meta:
        model = Cloth_S
        fields = ['photo']
