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
import jsonpickle
from json import JSONEncoder


@api_view(['POST'])
def post_cloth(request):
    print(request.data)
    # data = request.data.get("data")
    # data["photo"] = request.data.get("photo")
    # print(data["photo"])
    #decodedSet = jsonpickle.decode(request.data.get('season'))
    #request.data['season'] = decodedSet
    # print(decodedSet)
    decodedSet = jsonpickle.decode(request.data)
    print(decodedSet)
    print("start extract")
    extract_features(decodedSet)
    print("doen extract")
    # photo = request.data.get('photo')
    # print(request.data)
    return Response({'response': 'done'})


# @api_view(['POST'])
# def change_cloth(request):
#     #def extract_features_modify(request.data)
#     pass


@api_view(['POST'])
def delete_cloth(request):
    print(request.data)
    # data = request.data.get("data")
    # data["photo"] = request.data.get("photo")
    # print(data["photo"])
    delete_extract_features(request.data)
    # delete_extract_features(request.data)
    return Response({'response': 'done'})


@api_view(['POST'])
def modify_cloth(request):
    print(request.data)
    # data = request.data.get("data")
    # data["photo"] = request.data.get("photo")
    modify_extract_features(request.data)
    # modify_extract_features(request.data)
    return Response({'response': 'done'})


# @api_view(['POST'])
# def get_set(request):
#     #def extract_features_delete(request.data)
#     pass
