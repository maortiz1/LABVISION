import os
import requests
import zipfile
import tarfile

if not os.path.isdir(os.path.join(os.getcwd(),'wild')):
   url='https://drive.google.com/uc?export=download&id=1HYJCzhzdRb8m4yFjjuFxfUBam4IGrhg0'
   r=requests.get(url,allow_redirects=True)
   open('wild.zip','wb').write(r.content) 
   zip_ref = zipfile.ZipFile('wild.zip', 'r')
   zip_ref.extractall()
   zip_ref.close()

