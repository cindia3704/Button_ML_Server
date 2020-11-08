from django.shortcuts import render

# Create your views here.

# from django.shortcuts import render
# from rest_framework.parsers import JSONParser
# from rest_framework.decorators import api_view, permission_classes
from django.http import HttpResponse, JsonResponse
from rest_framework.response import Response
from rest_framework import status, views
from rest_framework.views import APIView
from . import polyvore
from .polyvore import run_inference
from .polyvore.run_inference import extract_features, delete_extract_features
from rest_framework.decorators import api_view, permission_classes
from .serializers import Cloth_SSerializer
import json
from .models import Cloth_S


@api_view(['POST'])
def post_cloth(request):
    photo = request.FILES["photo"]
    data = json.loads(request.data['data'])
    # print(photo)
    # print(data)
    # data["photo"] = photo
    # print(request.data.get('data'))
    #photo = request.data.get('photo')
    Cloth_S(photo=photo).save()
    print(photo)
    data_ = str(photo)
    print("data"+data_)
    #serializer = Cloth_SSerializer(data=photo)
    # print(serializer.is_valid())
    # if serializer.is_valid():
    print("inside")
    data['photo'] = data_
    print(data)
    extract_features(data)
    # photo = request.data.get('photo')
    # print(request.data)
    return Response({'response': 'done'})


# @api_view(['POST'])
# def change_cloth(request):
#     #def extract_features_modify(request.data)
#     pass


@api_view(['POST'])
def delete_cloth(request):
    delete_extract_features(request.data)
    return Response({'response': 'done'})


# @api_view(['POST'])
# def delete_cloth(request):
#     #def extract_features_delete(request.data)
#     pass


# @api_view(['POST'])
# def get_set(request):
#     #def extract_features_delete(request.data)
#     pass
