from django.shortcuts import render
from .models import *
from .forms import NewImageForm
from .forms import NewUserForm
from .forms import LatLngCityForm
from django.conf import settings
from django.conf.urls.static import static
from django.conf import settings
from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate, logout #add this #add this
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm #add this
from django.template import RequestContext
from datetime import datetime
#import settings
import openai
import requests
from PIL import Image
import requests

#openai.api_key ='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'



openai.api_key ='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'
#'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'
#uOIzsd0cmRGBSi8p631gT3BlbkFJc7y24Hex5TUzdsKFEXln'   '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'
#'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' #'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'

#from .models import AddLatLngCity
def galleryview(request):

    path = settings.MEDIA_ROOT
    img_list = os.listdir(path + "/images")
    context = {"images": img_list}
    return render (request, 'galleryview.html', context)
def listzonerecords(request):
    #zonerecords = LatLngCity.objects.all()
    #return render(request, 'listzonerecords.html', {'zonerecords':zonerecords})
   # dictionary for initial data with
    # field names as keys
    context ={}
    # add the dictionary during initialization
    context["dataset"] = LatLngCity.objects.all()
    return render(request, "listzonerecords.html", context)
  
from django.shortcuts import render
import csv
from pathlib import Path
import os
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
'''
def createsearch(request):
       # dictionary for initial data with
       # field names as keys
    if request.session.get('username')=='':
      form = AuthenticationForm()
      request.session['username'] = ''
      request.session['password'] = ''
      request.session['redirecttocreatesearch']='1'
      return render(request=request, template_name="login.html", context={"login_form":form})
    context ={}
    # add the dictionary during initialization
    form = NewSearchForm(request.POST or None)
    if form.is_valid():
        form.save()
        #response = HttpResponse(content_type='text/csv')  
        #response['Content-Disposition'] = 'attachment; #filename="file.csv"'  
        #writer = csv.writer(response)  
        UserInterest= request.POST['UserInterest']
        UserLocation= request.POST['UserLocation']
        #City = request.POST['City']
        #writer.writerow([Latitude,Longitude,City])  
        #textuserinterest = request.POST['UserInterest']
        #textuserlocation = request.POST['UserLocation']
        searchrecords= NewSearchModel.objects.all()
        GetRecommendations(UserInterest,UserLocation)
        context = {"searchrecords": searchrecords}
        return render (request, '/media/htmloutput.html', context)
    context['form']= form
    #form.fields.UserName.value='a'
    return render(request, "createsearch.html", context)
'''

def home(request):
  try:
    print (request.session['username'] )
  except:  
    request.session['username'] = ''
    request.session['password'] = ''
	

  return render(request, 'index.html',context_instance=RequestContext(request))

def createuserrecord(request):
    # dictionary for initial data with
    # field names as keys
    context ={}
    # add the dictionary during initialization
    form = NewUserForm(request.POST or None)
   
    if form.is_valid():
        form.save()
        #response = HttpResponse(content_type='text/csv')  
        #response['Content-Disposition'] = 'attachment; #filename="file.csv"'  
        #writer = csv.writer(response)  
        #Latitude= request.POST['Latitude']
        #Longitude= request.POST['Longitude']
        #City = request.POST['City']
        #writer.writerow([Latitude,Longitude,City])  
        #textuserinterest = request.POST['UserName']
        #textpassword= request.POST['UserPassword']
        users= NewUserModel.objects.all()
        context = {"users": users}
        return render (request, 'listuserrecords.html', context)
    context['form']= form
    return render(request, "createuserrecord.html", context)
    
def login_request(request):
    
	print('login')
	if request.method == "POST":
		form = AuthenticationForm(request, data=request.POST)
		if form.is_valid():
			username = form.cleaned_data.get('username')
			password = form.cleaned_data.get('password')
			user = authenticate(username=username, password=password)
			if user is not None:
				login(request, user)
				messages.info(request, f"You are now logged in as {username}.")
				request.session['username'] = username
				request.session['password'] = password   
				print(request.session['username'])
				try:
				  print (request.session['requestedform'])
				  form = NewImageForm(request.POST or None)
				  request.session['requestedform'] =''
				  context['form']= form
				  return render(request, "createimagerecord.html", context)
				except:
				  tmp=1
                
				"""
				if request.session['redirecttocreatesearch']='1':
                request.session['redirecttocreatesearch']=''
                return render(request, template_name="createsearch.html")
				"""  
				return render(request, template_name="index.html")
			else:
				messages.error(request,"Invalid username or password.")
		else:
			messages.error(request,"Invalid username or password.")
	form = AuthenticationForm()
	request.session['username'] = ''
	request.session['password'] = ''
	return render(request=request, template_name="login.html", context={"login_form":form})
    
def logout_request(request):
	print('logout')
	logout(request)
	messages.info(request, "You have successfully logged out.") 
	request.session['username']='Guest'
	request.session['password']=''
	return render(request, template_name="index.html")
def listuserrecords(request):
    users = User.objects.all()
    context = {"users": users}
    return render(request, 'listuserrecords.html', context)

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
def  createimagerecord(request):
  try:
    print(request.session['username'])
  except:
    request.session['username']=''
    request.session['password']=''
    if (request.session['username'] is None) or  (request.session['username'] =='') or (request.session['username'] =='Guest') :
      print(request.session['username']) 
      form = AuthenticationForm()
      request.session['username'] = ''
      request.session['password'] = ''
      request.session['requestedform'] = 'createimagerecord'
      return render(request=request, template_name="login.html", context={"login_form":form})
    # dictionary for initial data with
    # field names as keys
  context ={}
 
    # add the dictionary during initialization
  form = NewImageForm(request.POST or None)
  if form.is_valid():
        form.save()
        """response = HttpResponse(content_type='text/csv')  
        response['Content-Disposition'] = 'attachment; filename="file.csv"'  
        writer = csv.writer(response)  
        Latitude= request.POST['Latitude']
        Longitude= request.POST['Longitude']
        City = request.POST['City']
        writer.writerow([Latitude,Longitude,City])  """
        text = request.POST['ImageText']
        imgcount=1
        entrytime=request.POST['entrytime']
        GenerateImage(text,imgcount,entrytime)
        path =BASE_DIR# settings.MEDIA_ROOT
        #MEDIA_ROOT = os.path.join(BASE_DIR, 'media') # media directory in the root directory
        #MEDIA_URL = '/media/'
        img_list = "/media/pngimage1.png"
        img_list_varying=[]
        for i1 in range(0,imgcount):
           img_list_varying.append("/media/newoutput_" + str(i1) + ".png")
        
        #str(Path(__file__).resolve().parent.parent) + "\pngimage1.png"
        context = {"images": img_list,"imagesall":img_list_varying}
        return render (request, 'listimagerecords.html', context)
  context['form']= form
  context['username']= request.session['username']
  timenow = datetime.datetime.now()  
  context['entrytime']= timenow.strftime("%d-%m-%y %H %M %S")
  return render(request, "createimagerecord.html", context)
def listimagerecords(request):
   path =BASE_DIR# settings.MEDIA_ROOT
   #MEDIA_ROOT = os.path.join(BASE_DIR, 'media') # media directory in the root directory
   #MEDIA_URL = '/media/'
   imgcount=3
   img_list = "/media/pngimage1.png"
   img_list_varying=[]
   imagerecords = NewImageModel.objects.filter(username=request.session['username']).values()#all()
   for i1 in range(0,imgcount):
     img_list_varying.append("/media/newoutput_" + str(i1) + ".png")
        
   #str(Path(__file__).resolve().parent.parent) + "\pngimage1.png"
   
   imagerecordscount = NewImageModel.objects.filter(username=request.session['username']).values().count()
   context = {"imagerecords": imagerecords,"imagerecordscount":imagerecordscount,"images": img_list,"imagesall":img_list_varying ,"username":request.session["username"]}
   return render (request, 'listimagerecords.html', context)
    
def GenerateImage(text,imgcount,entrytime):

  url1 = generateimagefromai(text)
  url =url1
  response = requests.get(url)
  if response.status_code == 200:
    with open(settings.MEDIA_ROOT + "/sample.png", 'wb') as f:
        f.write(response.content)
  # opening the saved image and converting it into "RGBA" format
  # converted image is saved in result
  result = Image.open(settings.MEDIA_ROOT + '/sample.png').convert('RGBA')
  # saving the new image in PNG format
  result.save(settings.MEDIA_ROOT + '/img_rgba.png','PNG')
  #im=Image.open('img_rgba.png')
  #im.show()
  #-----------------------
  #To generate variations of an existing Image we use the “create_edit” endpoint of the DALL-E API.
  # editing image using create_edit endpoint of DALL-E API
  pngimage1 = Image.open(settings.MEDIA_ROOT + '/img_rgba.png').convert('RGBA')
  newsize = (256, 256)
  pngimage1 = pngimage1.resize(newsize)
  pngimage1.save(settings.MEDIA_ROOT + '/pngimage1.png','PNG')
  pngimage1.save(settings.MEDIA_ROOT + '/' + entrytime + '-pngimage1.png','PNG')
  pngmask = Image.open('mask.png').convert('RGBA')
  pngmask = pngmask.resize(newsize)
  pngmask.save(settings.MEDIA_ROOT + '/pngmask.png','PNG')
  #print('Enter desired image details:')
  inp='   '#input()
  response = openai.Image.create_edit(
  # opening original image in read mode
  image=open(settings.MEDIA_ROOT + "/pngimage1.png", "rb"),
  # opening mask image in read mode
  mask=open(settings.MEDIA_ROOT + "/pngmask.png", "rb"),
  # propmt describing the desired image
  #text=input()
  prompt=inp, #"gotham city skyline behind batman",
  # number of images to be generated
  n=imgcount,
  # size of each generated image
  size="256x256")
  
  # saving the URLs of all image in new variable "res"
  res = response['data']
  # loop to save and display images
  for i in range(len(res)):
    # saving URL of image in res
    image_url = res[i]['url']
    # extracting image from URL in bytes form
    response = requests.get(image_url, stream=True)
    # opening the image
    k = Image.open(response.raw)
    # displaying the image
    #k.show()
    # saving the image
    with open(settings.MEDIA_ROOT + '/' + f"img_variant_{i}.png", "wb") as f:
      f.write(response.content)
    with open(settings.MEDIA_ROOT + '/' +entrytime + "-" + f"img_variant_{i}.png", "wb") as f:
      f.write(response.content)
    
  for i in range(0,imgcount):
   response = openai.Image.create_variation(  image=open(settings.MEDIA_ROOT + "/pngimage1.png", "rb"),  n=1,  size="1024x1024")
   image_url = response['data'][0]['url']
   response = requests.get(image_url)
   if response.status_code == 200:
      with open(settings.MEDIA_ROOT + '/' + f"newoutput_{i}.png", 'wb') as f:
          f.write(response.content)
      with open(settings.MEDIA_ROOT + '/' +entrytime + "-" + f"newoutput_{i}.png", 'wb') as f:
          f.write(response.content)
def getfile(request):  
    response = HttpResponse(content_type='text/csv')  
    response['Content-Disposition'] = 'attachment; filename="file.csv"'  
    writer = csv.writer(response)  
    records = LatLngCity.objects.all()  
    writer = csv.writer(response)  
    for record in records:
      Latitude= record.Latitude
      Longitude= record.Longitude
      City = record.City
      Zone = record.Zone
      writer.writerow([Latitude,Longitude,City,Zone])  
    #writer.writerow(['1002', 'Amit', 'Mukharji', 'LA', '"Testing"'])  
    return response  
from geopy import distance 
#from django.contrib.gis.measure import D
#from django.contrib.gis.geos import * 
#from rest_framework import serializers
from django.core import serializers
from django.db import models
from django.utils import timezone

import sys

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import uuid, base64
from .models import *
from io import BytesIO
from matplotlib import pyplot

def generate_code():
    return str(uuid.uuid4()).replace('-', '').upper()[:12]
def get_key(res_by):
    if res_by == '#1':
        key = 'transaction_id'
    elif res_by == '#2':
        key = 'created'
    elif res_by == '#3':
        key = 'customer'
    elif res_by == '#4':
        key = 'total_price'
    return key
import numpy as np
def get_kmeanschart( kmeansdataframe):
    pyplot.switch_backend('AGG')
    fig = pyplot.figure(figsize=(10, 4))
    #key = get_key(results_by)
    df =kmeansdataframe#.groupby('City', as_index=False)['Zone'].agg('count')
    df_cluster = df[['Latitude', 'Longitude']]
    #X=np.array([[df_cluster.iloc[0]['Latitude'], df_cluster.iloc[0]['Longitude']],[df_cluster.iloc[1]['Latitude'], df_cluster.iloc[1]['Longitude']],[df_cluster.iloc[2]['Latitude'], df_cluster.iloc[2]['Longitude']], np.float64])
    #init=X
    kmeans = KMeans(n_clusters=3)#,init=X)
    kmeans.fit(df_cluster)
    
    #Getting the Centroids
    centroids = kmeans.cluster_centers_
    label = kmeans.fit_predict(df)
    u_labels = np.unique(label)
    #plotting the results:
    """
    for i in u_labels:
      plt.scatter(df[label == i , 0] , df[label == i , 1] )#, label = i)
      plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
      plt.legend()
    #plt.show()
    """
    dfclusters=[]
    for i in u_labels:
      dfclusters.append(df[label ==i])
    clr=['red','yellow','green']
    
    #for i in range(len(centroids)):
    #  plt.scatter(centroids[i,0], centroids[i,1], marker='x', s=400, linewidths=3,label = i)
    for i in u_labels:# range(len(dfclusters)):
      plt.scatter(dfclusters[i]['Latitude'],dfclusters[i]['Longitude'] ,c=clr[i],label = i)
      plt.scatter(centroids[i,0], centroids[i,1], marker='x', s=400, linewidths=3,c=clr[i])
  
    #plt.show()
    #kmeansdata.groupby(['Zone']).size().plot(kind = "bar")
    pyplot.tight_layout()
    buffer = BytesIO()
    pyplot.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    chart = graph
    return chart

def get_graph():
    buffer = BytesIO()
    pyplot.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph
def get_chart( data):
    pyplot.switch_backend('AGG')
    fig = pyplot.figure(figsize=(10, 4))
    #key = get_key(results_by)
    d =data#.groupby('City', as_index=False)['Zone'].agg('count')
    data.groupby(['Zone']).size().plot(kind = "bar")
    #if chart_type == '#1':
    #d=data.groupby('City',as_index=False)['Latitude'].agg('count')
    #print("Bar graph")
    
    #pyplot.plot(d['City'], d['Zone'], color='gray', marker='o', linestyle='dashed')
    
    #pyplot.plot(d['City'], d['Zone'], color='gray', marker='o', linestyle='dashed')
   
    pyplot.tight_layout()
    chart = get_graph()
    return chart

import pandas
from django.shortcuts import render
from django.views.generic import ListView
from django.contrib import messages
#from .forms import SalesSearchForm
from .models import *
# Create your views here.
#from .utils import get_chart

def zonewisecount(request):
    """
    sales_df = None
    chart = None
    no_data = None
    search_form = SalesSearchForm(request.POST or None)
    """
    
    qs = LatLngCity.objects.all()
    #df = pd.DataFrame({'Latitude': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],'Longitude': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24],'City': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A']})
    #df = pandas.DataFrame.from_records(qs.values())
    #datas = LatLngCity.objects.all()
    #datas = list(datas.values("City", "Latitude", "Longitude", "Zone"))
    #pd.DataFrame({'City': ['A', 'B'],'Latitude': [12, 20],'Longitude': [39, 36]})
    #df = pd.DataFrame(datas)#list(datas.values("City", "Latitude", "Longitude","Zone")))
    sales_df = pandas.DataFrame(qs.values())
    #salesdftable="india"
    chart = get_chart(sales_df)#, results_by)
    headers=convert_data_frame_to_html_table_headers(sales_df)
    rows=convert_data_frame_to_html_table_rows(sales_df)
    context = {'sales_df': sales_df,'chart': chart}
    return render(request, 'testgraph.html',  context)
def convert_data_frame_to_html_table_headers(df):  
  html = "<tr>"  
  for col in df.columns:  
    html += f"<th>{col}</th>"  
    html += "</tr>"  
  return html
      
def convert_data_frame_to_html_table_rows(df):   
  html = ""   
  for row in df.values:      
    row_html = "<tr>"      
    for value in row:         
      row_html += f"<td>{value}</td>"     
      row_html += "</tr>"
      html += row_html
  return html
def products(request):
    products = Product.objects.all()
    return render(request, 'products.html', {'products':products})
    
def customers(request):
    customers = Customer.objects.all()
    return render(request, 'customers.html', {'customers':customers})
# Create your views here.
def languages(request):
    languages = Language.objects.all()
    return render(request, 'languages.html', {'languages':languages})
# Create your views here.

from django.http import HttpResponse
# get datetime
import datetime
 
from django.shortcuts import render
 
# relative import of forms
from .models import GeeksModel
from .forms import GeeksForm
from django.shortcuts import render
def create_view(request):
    # dictionary for initial data with
    # field names as keys
    context ={}
 
    # add the dictionary during initialization
    form = GeeksForm(request.POST or None)
    if form.is_valid():
        form.save()
        return HttpResponseRedirect("/list_view")
    context['form']= form
    return render(request, "create_view.html", context)
    
    
 
# relative import of forms
from .models import GeeksModel
 
 
def list_view(request):
    # dictionary for initial data with
    # field names as keys
    context ={}
 
    # add the dictionary during initialization
    context["dataset"] = GeeksModel.objects.all()
         
    return render(request, "list_view.html", context)
    
from django.urls import path
from django.shortcuts import render
 
# relative import of forms
 
# pass id attribute from urls
def detail_view(request, id):
    # dictionary for initial data with
    # field names as keys
    context ={}
 
    # add the dictionary during initialization
    context["data"] = GeeksModel.objects.get(id = id)
         
    return render(request, "detail_view.html", context)
    
    from django.shortcuts import (get_object_or_404,
                              render,
                              HttpResponseRedirect)

from django.shortcuts import (get_object_or_404,
                              render,
                              HttpResponseRedirect)
 
# relative import of forms
from .models import GeeksModel
from .forms import GeeksForm
# update view for details
def update_view(request, id):
    # dictionary for initial data with
    # field names as keys
    context ={}
 
    # fetch the object related to passed id
    obj = get_object_or_404(GeeksModel, id = id)
 
    # pass the object as instance in form
    form = GeeksForm(request.POST or None, instance = obj)
 
    # save the data from the form and
    # redirect to detail_view
    if form.is_valid():
        form.save()
        return HttpResponseRedirect("/"+id)
 
    # add form dictionary to context
    context["form"] = form
 
    return render(request, "update_view.html", context)
    
    from django.shortcuts import (get_object_or_404,
                              render,
                              HttpResponseRedirect)
 
from .models import GeeksModel
 
 
# delete view for details
def delete_view(request, id):
    # dictionary for initial data with
    # field names as keys
    context ={}
 
    # fetch the object related to passed id
    obj = get_object_or_404(GeeksModel, id = id)
 
 
    if request.method =="POST":
        # delete object
        obj.delete()
        # after deleting redirect to
        # home page
        return HttpResponseRedirect("/list_view")
 
    return render(request, "delete_view.html", context)
    
from django.core.files.storage import FileSystemStorage
def fileupload(request):
 if request.method == "POST":
    # if the post request has a file under the input name 'document', then save the file.
    request_file = request.FILES['document'] if 'document' in request.FILES else None
    if request_file:
            # save attached file
 
            # create a new instance of FileSystemStorage
            fs = FileSystemStorage()
            file = fs.save(request_file.name, request_file)
            # the fileurl variable now contains the url to the file. This can be used to serve the file when needed.
            fileurl = fs.url(file)
 
 return render(request, "fileupload.html")
 
 
 #
 #https://www.geeksforgeeks.org/generate-images-with-openai-in-python/

#softproms@gmail.com
#openai.com Personal -> View API Keys ->Create new key->MyAIImageGenerator->Create

#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

# importing openai module
# assigning API KEY to the variable
#keys generated from here https://platform.openai.com  
 #'API_KEY'
# importing other libraries
#------------------
#Note: The size of the generated images must be one of 256×256, 512×512, or 1024×1024.
# function for text-to-image generation 
# using create endpoint of DALL-E API
# function takes in a string argument
def generateimagefromai(text):
  res = openai.Image.create(
    # text describing the generated image
    prompt=text,
    # number of images to generate 
    n=1,
    # size of each generated image
    size="256x256",
  )
  # returning the URL of one image as 
  # we are generating only one image
  return res["data"][0]["url"]

"""
im=Image.open('pngimage1.png')
im.show()
"""
