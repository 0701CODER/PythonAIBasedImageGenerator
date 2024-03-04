#https://www.geeksforgeeks.org/generate-images-with-openai-in-python/

#softproms@gmail.com
#openai.com Personal -> View API Keys ->Create new key->MyAIImageGenerator->Create

#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

# importing openai module
import openai
# assigning API KEY to the variable
#keys generated from here https://platform.openai.com  
openai.api_key ='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' #'API_KEY'

# importing other libraries
import requests
from PIL import Image



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

import requests
# prompt describing the desired image
print('Enter desired image details:')
text=input()
#text = "batman art in red and blue color"
# calling the custom function "generate"
# saving the output in "url1"
url1 = generate(text)
#print(url1)
#exit()
# using requests library to get the image in bytes
#response = requests.get(url1)
url =url1
response = requests.get(url)
if response.status_code == 200:
    with open("sample.png", 'wb') as f:
        f.write(response.content)

# using the Image module from PIL library to view the image
#print(response.raw)
#Image.open(response.raw)
#img = response.raw.read()
#print(img)
#with open('test1.jpg', 'wb') as f:
#  f.write(img)
#exit()
#----------------------
#How to Generate Variations of an Image?
#Here we are going to use the same image generated above by DALL E and generate its variations.
#Since DALL E only accepts square PNG images with sizes less than 4 MB and in RGBA format, we save our image with extension png and in RGBA format using the following code.
"""
response = requests.get(url1)
# saving the image in PNG format
with open("img.png", "wb") as f:
  f.write(response.content)
"""  
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
  size="256x256"
)
"""
response = openai.Image.create_edit(
  # opening original image in read mode
  image=open("img_rgba.png", "rb"),
  # opening mask image in read mode
  mask=open("mask.png", "rb"),
  # propmt describing the desired image
  print('Enter desired image details:')
  #text=input()
  prompt=input() #"gotham city skyline behind batman",
  # number of images to be generated
  n=3,
  # size of each generated image
  size="256x256"
)

"""
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

"""
for i in range(len(res)):
  im=Image.open(f"img_variant_{i}.png")
  im.show()
"""    
    
for i in range(0,10):
 response = openai.Image.create_variation(  image=open("pngimage1.png", "rb"),  n=1,  size="1024x1024")
 image_url = response['data'][0]['url']
 response = requests.get(image_url)
 if response.status_code == 200:
    with open(f"newoutput_{i}.png", 'wb') as f:
        f.write(response.content)
"""
im=Image.open('pngimage1.png')
im.show()
"""
exit()
    
#-----------------------
