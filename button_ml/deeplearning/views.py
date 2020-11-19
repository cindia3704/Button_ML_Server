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
from .polyvore import run_inference, set_generation
from .polyvore.set_generation import set_generation
from .polyvore.run_inference import extract_features, delete_extract_features, modify_extract_features
from rest_framework.decorators import api_view, permission_classes
from .serializers import Cloth_SSerializer
import json
from .models import Cloth_S
import jsonpickle
from json import JSONEncoder
import requests
from django.http import JsonResponse


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
    decodedSet = jsonpickle.decode(request.data)
    print(decodedSet)
    print("start del extract")
    delete_extract_features(decodedSet)
    print("doen del extract")

    # delete_extract_features(request.data)
    return Response({'response': 'done'})


@api_view(['POST'])
def modify_cloth(request):
    print(request.data)
    # data = request.data.get("data")
    # data["photo"] = request.data.get("photo")
    decodedSet = jsonpickle.decode(request.data)
    print(decodedSet)
    print("start mod extract")
    modify_extract_features(decodedSet)
    # delete_extract_features(decodedSet)
    print("doen mod extract")
    # modify_extract_features(request.data)
    return Response({'response': 'done'})


@api_view(['GET'])
def get_set(request):
    decodedSet = jsonpickle.decode(request.data)
    print(decodedSet)
    bi_lstm_input = decodedSet['bi_lstm_input']
    id = decodedSet['id']
    style = decodedSet['style']
    season = decodedSet['season']
    cloth_set = set_generation(bi_lstm_input, id, style, season)
    print("cloth_set")
    print(cloth_set)
    print(len(cloth_set))
    clo = []
    for clothes in cloth_set:
        clo.append(str(clothes))
    # send_data = {
    #     "id": id,
    #     "rand_cloth": bi_lstm_input,
    #     "season": season,
    #     'clothlist': cloth_set
    # }
    # encoded = jsonpickle.encode(send_data)
    # r = requests.post(
    #     'https://capstonebutton.kro.kr:9000/getSetRec/', json=encoded)
    print("cloth_set:")
    print(cloth_set)
    send_data = {
        'clothlist': cloth_set
    }
    encoded = jsonpickle.encode(send_data)
    return JsonResponse(send_data, safe=False)
