from django.db import models
"""
class NewImageModel(models.Model):
    ImageText = models.CharField(max_length=200)
    #Latitude = models.DecimalField(max_digits=10, decimal_places=2)
    #Longitude = models.DecimalField(max_digits=10, decimal_places=6)
    #Longitude = models.CharField(max_length=200)
    #City = models.CharField(max_length=200)
    #COLOR_CHOICES=(    ('Unknown','Unknown'),    ('Green','Green'),   ('Red','Red'),    ('Blue','Blue'))
    #Zone = models.CharField(max_length=20,choices=COLOR_CHOICES, default='Green')
    #description = models.TextField()
    #image = models.CharField(max_length=5000, null=True, blank=True)
    def __str__(self):
        return self.title
"""
"""
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
    def __str__(self):
        return self.title
"""