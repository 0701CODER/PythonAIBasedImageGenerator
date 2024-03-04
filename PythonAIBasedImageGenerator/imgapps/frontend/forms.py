from django import forms
from .models import GeeksModel
from .models import NewUserModel
from .models import NewImageModel
from .models import LatLngCity

from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
class NewUserForm(UserCreationForm):
	email = forms.EmailField(required=True)

	class Meta:
		model = User
		fields = ("username", "email", "password1", "password2")

	def save(self, commit=True):
		user = super(NewUserForm, self).save(commit=False)
		user.email = self.cleaned_data['email']
		if commit:
			user.save()
		return user
class NewImageForm(forms.ModelForm):
    # create meta class
    class Meta:
        # specify model to be used
        model = NewImageModel
 
        # specify fields to be used
        fields = [
            "ImageText",
            "username",
            "entrytime",
        ]

 # creating a form
class LatLngCityForm(forms.ModelForm):
 
    # create meta class
    class Meta:
        # specify model to be used
        model = LatLngCity
 
        # specify fields to be used
        fields = [
            "Latitude",
            "Longitude",
            "City",
            "Zone",
        ]
# creating a form
class GeeksForm(forms.ModelForm):
 
    # create meta class
    class Meta:
        # specify model to be used
        model = GeeksModel
 
        # specify fields to be used
        fields = [
            "title",
            "description",
        ]