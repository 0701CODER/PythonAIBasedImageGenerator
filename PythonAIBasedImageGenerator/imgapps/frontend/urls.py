from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views
from frontend.views import listzonerecords
from frontend.views import createimagerecord
#from frontend.views import CalculateLabelZones
#from frontend.views import geeks_view
#from frontend.views import hello
from frontend.views import home
from frontend.views import createuserrecord
from frontend.views import listuserrecords
from frontend.views import create_view
from frontend.views import galleryview
from frontend.views import list_view
from frontend.views import detail_view
from frontend.views import update_view
from frontend.views import delete_view
from frontend.views import fileupload
from frontend.views import zonewisecount
from django.conf import settings
from django.conf.urls.static import static
#from frontend.views import getfileproposed
#from frontend.views import CalculateLabelZonesByLatLng


urlpatterns = [
    path('', views.home, name='home'),
    path('createimagerecord/', views.createimagerecord, name='createimagerecord'),
    path('galleryview/', views.galleryview, name='listzonerecords'),
    path('listimagerecords/', views.listimagerecords, name='listimagerecords'),
    path('createuserrecord/', views.createuserrecord, name='createuserrecord'),
    path('listuserrecords/', views.listuserrecords, name='listuserrecords'),
    path('login/', views.login_request, name='login'),
    path('/login', views.login_request, name='login'),
    path("logout", views.logout_request, name= "logout"),
    

path('zonewisecount/',zonewisecount),
]   + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
#path('detail_view/', list_view),
#path('calculatelabelzones/',CalculateLabelZones),
#    path('CalculateLabelZonesByLatLng/',CalculateLabelZonesByLatLng),
#path('geeks/', geeks_view),
#path('hello/', hello),
#path('csvproposed/',views.getfileproposed)  ,
"""
 path('products/', views.products, name='products'),
    
    path('customers/', views.customers, name='customers'),
    
    path('listcustomers/', views.listcustomers, name='listcustomers'),
    
    
    path('create_view/', create_view),
    path('list_view/', list_view),
    path('details_view/<id>', detail_view ),
    path('update_view/<id>', update_view ),
    path('<id>/delete', delete_view ),
    path('', views.home,name="home"),
    path('languages/', views.languages, name='languages'),
    path('fileupload/', views.fileupload, name='fileupload'),
    path('csv',views.getfile)  ,
"""