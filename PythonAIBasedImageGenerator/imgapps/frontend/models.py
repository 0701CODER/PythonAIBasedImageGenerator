from django.db import models
from django  import forms
#from django.forms import *
#from django  import widget
from django.shortcuts import render
class NewImageModel(models.Model):
    ImageText = models.CharField(max_length=200)
    username= models.CharField(max_length=20,default='')
    entrytime= models.CharField(max_length=20,default='')
    #Latitude = models.DecimalField(max_digits=10, decimal_places=2)
    #Longitude = models.DecimalField(max_digits=10, decimal_places=6)
    #Longitude = models.CharField(max_length=200)
    #City = models.CharField(max_length=200)
    #COLOR_CHOICES=(    ('Unknown','Unknown'),    ('Green','Green'),   ('Red','Red'),    ('Blue','Blue'))
    #Zone = models.CharField(max_length=20,choices=COLOR_CHOICES, default='Green')
    #description = models.TextField()
    #image = models.CharField(max_length=5000, null=True, blank=True)
    '''def __str__(self):
        return ''#self.title
    '''

class NewUserModel(models.Model):
    #CHOICES=(    ('wildlife','wildlife'),    ('heritage','heritage'),   ('pilgrimage','pilgrimage'),    ('park','park'),    ('museum','museum'))
    
    UserName= models.CharField(max_length=20,default='',primary_key = True)
    Password = models.CharField(max_length=20,default='')
    #Password = models.CharField(widget=forms.PasswordInput,max_length=20)
    #Latitude = models.DecimalField(max_digits=10, decimal_places=2)
    #Longitude = models.DecimalField(max_digits=10, decimal_places=6)
    #Longitude = models.CharField(max_length=200)
    #City = models.CharField(max_length=200)
    
    #Zone = models.CharField(max_length=20,choices=COLOR_CHOICES, default='Green')
    #description = models.TextField()
    #image = models.CharField(max_length=5000, null=True, blank=True)
    def __str__(self):
        return self.title
class LatLngCity(models.Model):
    Latitude = models.DecimalField(max_digits=10, decimal_places=6)
    #Latitude = models.DecimalField(max_digits=10, decimal_places=2)
    Longitude = models.DecimalField(max_digits=10, decimal_places=6)
    #Longitude = models.CharField(max_length=200)
    City = models.CharField(max_length=200)
    COLOR_CHOICES=(
    ('Unknown','Unknown'),
    ('Green','Green'),
    ('Red','Red'),
    ('Blue','Blue'))
    Zone = models.CharField(max_length=20,choices=COLOR_CHOICES, default='Green')
    #description = models.TextField()
    #image = models.CharField(max_length=5000, null=True, blank=True)
    def __str__(self):
        return self.title
class Product(models.Model):
    productname = models.CharField(max_length=200)
    price = models.DecimalField(max_digits=5, decimal_places=2)
    description = models.TextField()
    image = models.CharField(max_length=5000, null=True, blank=True)

     
# declare a new model with a name "GeeksModel"
class Customer(models.Model):
    # fields of the model
    customerid= models.CharField(max_length = 200)
    customername = models.TextField()
    # renames the instances of the model
    # with their title name
    """
    def __str__(self):
        return self.title
    """

  
# declare a new model with a name "GeeksModel"
class GeeksModel(models.Model):
 
    # fields of the model
    title = models.CharField(max_length = 200)
    description = models.TextField()
 
    # renames the instances of the model
    # with their title name
    def __str__(self):
        return self.title
        
from django.db import models
  
# Create your models here.
class Language(models.Model):
    name = models.CharField(max_length=20)
  
    def __str__(self):
        return f"{self.name}"
        
        from django.shortcuts import render
from .models import Language
  
# Create your views here.
def home(request):
    languages = Language.objects.all()
    return render(request,'index.html',{"languages":languages})
    
    
# Create your models here.
class LanguageTemp(models.Model):
    name = models.CharField(max_length=20)
  
    def __str__(self):
        return f"{self.name}"
        
        from django.shortcuts import render

 