
import os
import requests
import urllib

URL='https://www.dropbox.com/s/bijwrvojafs2nwn/checkpoint.pth.tar?dl=1'
print('It will be proceed to download the model')
#checking if databse is already downloaded
if not(os.path.exists('checkpoint.pth.tar')):
  urllib.request.urlretrieve(URL, "checkpoint.pth.tar") 
  print('The model had been downloaded')
else: 
  print('The file  already exists')


