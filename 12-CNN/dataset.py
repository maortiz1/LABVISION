


class Datset(path)
    def __init__(self,path):
    
     URL='https://www.dropbox.com/s/pcyp3a2an1shj5c/celeba-dataset.zip?dl=1'
     print('It will be proceed to download the database')
#checking if databse is already downloaded
     if not(os.path.exists('celeba-dataset.zip')):
        urllib.request.urlretrieve(URL, "celeba-dataset.zip") 
        print('The database had been download')
     else: 
        print('The file celeba-dataset.zip already exists')
    
     print('It will be proceed to decompress de database')
#checking if database is already decompress

     if not(os.path.exists('celeba-dataset')):
        zips = zipfile.ZipFile('celeba-dataset.zip','r')
        zips.extractall('celeba-dataset')
        zips.close()
     if not(os.path.exists('celeba-dataset/img_align_celeba')):
        zip1 = zipfile.ZipFile('celeba-dataset/img_align_celeba.zip','r')
        print('Unzipping part1')
        zip1.extractall('celeba-dataset/img_align_celeba')
        zip1.close()
     
    
    