import ipdb 
import requests    
import tarfile
import os
import zipfile

if not os.path.isdir(os.path.join(os.getcwd(),'ISIC2018_Task1-2_Training_Input')):
   print('It will be proceed to download the training database')
   url='https://challenge.kitware.com/api/v1/item/5ac37a9d56357d4ff856e176/download'
   r=requests.get(url,allow_redirects=True)
   open('ISIC2018_Task1-2_Training_Input.zip','wb').write(r.content) 
   print('The training database had been download')
   print('It will be proceed to decompress de training database')
   zip_ref = zipfile.ZipFile('ISIC2018_Task1-2_Training_Input.zip', 'r')
   zip_ref.extractall()
   zip_ref.close()
   
   
   

url ='https://challenge.kitware.com/api/v1/item/5ac3695656357d4ff856e16a/download'


if not os.path.isdir(os.path.join(os.getcwd(),'ISIC2018_Task1_Training_GroundTruth')):
   print('It will be proceed to download the training database  groundtruht')
   url='https://challenge.kitware.com/api/v1/item/5ac3695656357d4ff856e16a/download'
   r=requests.get(url,allow_redirects=True)
   open('ISIC2018_Task1_Training_GroundTruth.zip','wb').write(r.content) 
   print('The training database had been download')
   print('It will be proceed to decompress de training database')
   zip_ref = zipfile.ZipFile('ISIC2018_Task1_Training_GroundTruth.zip', 'r')
   zip_ref.extractall()
   zip_ref.close()