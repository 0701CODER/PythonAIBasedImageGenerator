from django.shortcuts import render
from .models import *
from .forms import NewImageForm
from .forms import LatLngCityForm
#from .models import AddLatLngCity
def home(request):
  return render(request, 'index.html')
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
def  createimagerecord(request):
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
    #writer.writerow(['1002', 'Amit', 'Mukharji', 'LA', '"Testing"'])  
        #return response  
        #getfile(request)
        # Create the HttpResponse object with the appropriate CSV header.
        #response = HttpResponse('text/csv')
        #response['Content-Disposition'] = 'attachment; filename=quiz.csv'
        # Create the CSV writer using the HttpResponse as the "file"
        #writer = csv.writer(response)
        #writer.writerow(['Student Name', 'Quiz Subject'])
        """
        try:
          print("try") 
          if 'myLatLngForm' in request.POST: #.get('f1'):
            Latitude= request.POST['Latitude']
            Longitude= request.POST['Longitude']
            City = request.POST['City']
            with open("{% static 'csv/latlngdata.csv' %}", 'a') as file:
              myFile = csv.writer(file)
              # myFile.writerow(["Id", "Activity", "Description", "Level"])
              myFile.writerow( [Latitude,Longitude,City])
              print('Written')
        except Exception as e:  
            print('file error :', e)
        """    
        
        GenerateImage():
        
        
        return HttpResponseRedirect("/listimagerecords")
    context['form']= form
    return render(request, "createimagerecord.html", context)
def GenerateImage(text):
  url1 = generate(text)
  url =url1
  response = requests.get(url)
  if response.status_code == 200:
    with open("sample.png", 'wb') as f:
        f.write(response.content)
  # opening the saved image and converting it into "RGBA" format
  # converted image is saved in result
  result = Image.open('sample.png').convert('RGBA')
  # saving the new image in PNG format
  result.save('img_rgba.png','PNG')
  im=Image.open('img_rgba.png')
  im.show()
  #-----------------------
  #To generate variations of an existing Image we use the “create_edit” endpoint of the DALL-E API.
  # editing image using create_edit endpoint of DALL-E API
  pngimage1 = Image.open('img_rgba.png').convert('RGBA')
  newsize = (256, 256)
  pngimage1 = pngimage1.resize(newsize)
  pngimage1.save('pngimage1.png','PNG')
  pngmask = Image.open('mask.png').convert('RGBA')
  pngmask = pngmask.resize(newsize)
  pngmask.save('pngmask.png','PNG')
  #print('Enter desired image details:')
  inp='   '#input()
  response = openai.Image.create_edit(
  # opening original image in read mode
  image=open("pngimage1.png", "rb"),
  # opening mask image in read mode
  mask=open("pngmask.png", "rb"),
  # propmt describing the desired image
  #text=input()
  prompt=inp, #"gotham city skyline behind batman",
  # number of images to be generated
  n=3,
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
    k.show()
    # saving the image
    with open(f"img_variant_{i}.png", "wb") as f:
      f.write(response.content)

    
  for i in range(0,10):
   response = openai.Image.create_variation(  image=open("pngimage1.png", "rb"),  n=1,  size="1024x1024")
   image_url = response['data'][0]['url']
   response = requests.get(image_url)
   if response.status_code == 200:
      with open(f"newoutput_{i}.png", 'wb') as f:
          f.write(response.content)
"""
from math import sin, cos, sqrt, atan2, radians
def FindDistanceByLatLng(lat1,lng1,lat2,lng2):
  # Approximate radius of earth in km
  R = 6373.0
  lat1 = radians(lat1)
  lon1 = radians(lng1)
  lat2 = radians(lat2)
  lon2 = radians(lng2)
  dlon = lon2 - lon1
  dlat = lat2 - lat1
  a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
  c = 2 * atan2(sqrt(a), sqrt(1 - a))
  distance = R * c
  distance=round(distance,3)
  return distance
  print("Result: ", distance)
  #print("Should be: ", 278.546, "km")
def CalculateLabelZonesByLatLng(request):
  records=LatLngCity.objects.all()
  records = serializers.serialize("python", LatLngCity.objects.all())

  #for object in records:
  #  print (object.Latitude)#field_name, field_value
        #Convert this records into dataframe as find nearest city distancewise

  qs = LatLngCity.objects.all()#select_related().filter(date__year=2012)
  q = qs.values('Latitude', 'Longitude','City','Zone')
  df = pd.DataFrame.from_records(q)
  
  lat1 = df.iloc[0,0]  #Red Zone
  lng1 = df.iloc[0,1]
  lat2 = df.iloc[1,0]  #Yellow Zone
  lng2 = df.iloc[1,1]
  lat3 = df.iloc[2,0]  #Green Zone
  lng3 = df.iloc[2,1]
  
  leng =len(df)
  distances=[]
  snos=[]
  distFromR=0
  distFromY=0
  distFromG=0
  
  zones=[]
  for i in range(3,leng):
    lati1 = df.iloc[i,0]
    longi1 = df.iloc[i,1]
    city=df.iloc[i,2]
    distFromR=0
    distFromY=0
    distFromG=0
    distFromR =FindDistanceByLatLng(lat1,lng1,lati1,longi1)
    distFromY =FindDistanceByLatLng(lat2,lng2,lati1,longi1)
    distFromG =FindDistanceByLatLng(lat3,lng3,lati1,longi1)
    zone='Red'
    dist=0
    if (distFromR <distFromY and distFromR <distFromG):
      zone ='Red'
      dist=distFromR
    if (distFromY <distFromR and distFromY <distFromG):
      zone ='Yellow'  
      dist=distFromY
    if (distFromG <distFromR and distFromG <distFromY):
      zone ='Green' 
      dist=distFromG
    alldist = str(distFromR) + " " +  str(distFromY) + " " +  str(distFromG) 
    snos.append({'SNo':str(i+1),'Zone':zone,'Dist':dist,'AllDistances':alldist,'City':city,'Latitude':lati1,'Longitude':longi1})
    zones.append(zone)
    distances.append(dist)
    #print(distances[i-3])
  q = qs.values('Latitude', 'Longitude')
  df = pd.DataFrame.from_records(q)
  chart=  get_kmeanschart(df)
  dist= FindDistanceByLatLng(lat1,lng1,lat2,lng2)
  context = {'distances':distances,'snos':snos}
  #context = {'chart': chart,'dist':dist}
  return render(request, 'calculatezonedetails.html',  context)
"""  
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
def getfileproposed(request):  
  records=LatLngCity.objects.all()
  records = serializers.serialize("python", LatLngCity.objects.all())

  #for object in records:
  #  print (object.Latitude)#field_name, field_value
        #Convert this records into dataframe as find nearest city distancewise

  qs = LatLngCity.objects.all()#select_related().filter(date__year=2012)
  q = qs.values('Latitude', 'Longitude','City','Zone')
  df = pd.DataFrame.from_records(q)
  
  lat1 = df.iloc[0,0]  #Red Zone
  lng1 = df.iloc[0,1]
  lat2 = df.iloc[1,0]  #Yellow Zone
  lng2 = df.iloc[1,1]
  lat3 = df.iloc[2,0]  #Green Zone
  lng3 = df.iloc[2,1]
  
  leng =len(df)
  distances=[]
  snos=[]
  distFromR=0
  distFromY=0
  distFromG=0
  
  zones=[]
  response = HttpResponse(content_type='text/csv')  
  response['Content-Disposition'] = 'attachment; filename="fileproposed.csv"'  
  writer = csv.writer(response)  
  #records = LatLngCity.objects.all()  
  #writer = csv.writer(response)  
  records=snos
  str1='SNo,City,Zone,Dist,Location'
  writer.writerow([str1])
  for i in range(3,leng):
    lati1 = df.iloc[i,0]
    longi1 = df.iloc[i,1]
    city=df.iloc[i,2]
    distFromR=0
    distFromY=0
    distFromG=0
    distFromR =FindDistanceByLatLng(lat1,lng1,lati1,longi1)
    distFromY =FindDistanceByLatLng(lat2,lng2,lati1,longi1)
    distFromG =FindDistanceByLatLng(lat3,lng3,lati1,longi1)
    zone='Red'
    dist=0
    if (distFromR <distFromY and distFromR <distFromG):
      zone ='Red'
      dist=distFromR
    if (distFromY <distFromR and distFromY <distFromG):
      zone ='Yellow'  
      dist=distFromY
    if (distFromG <distFromR and distFromG <distFromY):
      zone ='Green' 
      dist=distFromG
    alldist = str(distFromR) + " " +  str(distFromY) + " " +  str(distFromG) 
    snos.append({'SNo':str(i+1),'Zone':zone,'Dist':dist,'AllDistances':alldist,'City':city,'Latitude':lati1,'Longitude':longi1})
    str1=str(i+1) + ","  + city + "," + zone  + "," + str(dist) + "," + str(lati1) + " " +str(longi1) +"\r\n"
    writer.writerow([str(i+1),city,zone,dist,str(lati1),str(longi1)])
    zones.append(zone)
    distances.append(dist)
    #print(distances[i-3])

  

  #for sno in snos:
  #    writer.writerow(sno)  
  """
  for record in records:
      Latitude= record.Latitude
      Longitude= record.Longitude
      City = record.City
      Zone = record.Zone
      writer.writerow([Latitude,Longitude,City,Zone])  
    #writer.writerow(['1002', 'Amit', 'Mukharji', 'LA', '"Testing"'])  
  """
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
def CalculateLabelZones(request):
  records=LatLngCity.objects.all()
  records = serializers.serialize("python", LatLngCity.objects.all())

  #for object in records:
  #      print (object.Latitude)#field_name, field_value
        #Convert this records into dataframe as find nearest city distancewise

  qs = LatLngCity.objects.all()#select_related().filter(date__year=2012)
  q = qs.values('Latitude', 'Longitude')
  df = pd.DataFrame.from_records(q)
  chart=  get_kmeanschart(df)
  context = {'chart': chart}
  return render(request, 'kmeanschart.html',  context)
#df = pd.read_csv('your_data.csv')
  """
  df_cluster = df[['Latitude', 'Longitude']]
  kmeans = KMeans(n_clusters=3)
  kmeans.fit(df_cluster)
  plt.scatter(df_cluster['Latitude'], df_cluster['Longitude'], c=kmeans.labels_, cmap='viridis')
  centers = kmeans.cluster_centers_
  plt.scatter(centers[:, 0], centers[:, 1], c='red', s=300, alpha=0.5)
#plt.show()
  plt.savefig("output.jpg")
  #LatLngCity.objects.filter(id=field_value).update(Zone='BBB')
  #pnt = fromstr('POINT(48.796777 2.371140 )', srid=4326)
  #qs = MyResult.objects.filter(point__distance_lte=(pnt, D(km=20)))
  #_, ne = g.geocode('Newport, RI')  
  #_, cl = g.geocode('Cleveland, OH')  
  #distance.distance(ne, cl).miles  

  #for rec in records:
  #   lat=rec["Latitude"]
  #LatLngCity.objects.filter(id=11).update(Zone='AAA')
  #obj = get_object_or_404(LatLngCity, id = 11)
  """
  return HttpResponseRedirect("/kmeansoutput")
# Create your models here.
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
    
    """
    plt.scatter(df_cluster['Latitude'], df_cluster['Longitude'], c=kmeans.labels_, cmap='viridis')
    #plt.scatter(df_cluster['Latitude'], df_cluster['Longitude'], c=clr, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c=['r','g','b'], s=300, alpha=0.5)
    """
    
    """for i in range(centers):
    plt.scatter(data2D[labels==i,0], data2D[labels==i,1], c=colors[i])
    plt.scatter(centers2D[i,0], centers2D[i,1], c=colors[i], marker='x', s=200, linewidths=2)
    """
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
    """elif chart_type == '#2':
        print("Pie chart")
        pyplot.pie(data=d,x='total_price', labels=d[key])
    elif chart_type == '#3':
        print("Line graph")
        pyplot.plot(d[key], d['total_price'], color='gray', marker='o', linestyle='dashed')
    else:
        print("Apparently...chart_type not identified")
    """
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

"""
def listLatLngRecords(request):  
    context ={}
 
    # add the dictionary during initialization
    context["dataset"] = LatLngCity.objects.all()
         
    return render(request, "listcustomers.html", context)
"""

def listcustomers(request):
    # dictionary for initial data with
    # field names as keys
    context ={}
 
    # add the dictionary during initialization
    context["dataset"] = Customer.objects.all()
         
    return render(request, "listcustomers.html", context)

from django.http import HttpResponse
# get datetime
import datetime
 
# create a function
def geeks_view(request):
    # fetch date and time
    now = datetime.datetime.now()
    # convert to string
    #html = "<html><body>"
    #html += "Welcome<br/>"
    html = "Time is {}".format(now) 
    html+= "</body></html>"
    # return response
    return HttpResponse(html)
    
    
# create a function
def hello(request):
    # fetch date and time
    now = datetime.datetime.now()
    # convert to string
    html = "Hello, Time is {}".format(now)
    # return response
    return HttpResponse(html)
    
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
import openai
import requests
from PIL import Image
import requests
# assigning API KEY to the variable
#keys generated from here https://platform.openai.com  
openai.api_key ='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' #'API_KEY'
# importing other libraries



#------------------
#Note: The size of the generated images must be one of 256×256, 512×512, or 1024×1024.
# function for text-to-image generation 
# using create endpoint of DALL-E API
# function takes in a string argument
def generate(text):
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