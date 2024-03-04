
https://www.geeksforgeeks.org/django-project-creating-a-basic-e-commerce-website-for-displaying-products/?ref=ml_lbp




how to create superuser
------------------------
python manage.py createsuperuser
https://www.educative.io/answers/how-to-create-a-superuser-in-django




In 
http://127.0.0.1:8000/admin/
we need to give that username and password


1. we can add records in admin panel
2. this method also

C:\Users\Training\ecom>python manage.py shell
Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
(InteractiveConsole)
>>> from frontend.models import Customer
>>> Customer.objects.create(customerid="3", customername="radha").save()
>>> Customer.objects.create(customerid="4", customername="radhika").save()
>>>




