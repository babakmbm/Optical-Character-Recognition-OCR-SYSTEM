import pandas as pd
import xml.etree.ElementTree as xet
from glob import glob

path = glob('*.xml')
#print(path)

# Pars xml information

#Create a dictionary

labels_dict = dict(filepath = [],xmin = [],ymin =[],xmax=[],ymax=[])
for filename in path:
    #filename = path[0]
    info = xet.parse(filename)
    root = info.getroot()
    member_object = root.find('object')
    labels_info = member_object.find('bndbox')
    xmin = int(labels_info.find('xmin').text)
    ymin = int(labels_info.find('ymin').text)
    xmax = int(labels_info.find('xmax').text)
    ymax = int(labels_info.find('ymax').text)
    #print(xmin,ymin,xmax,ymax)
    labels_dict['filepath'].append(filename)
    labels_dict['xmin'].append(xmin)
    labels_dict['ymin'].append(ymin)
    labels_dict['xmax'].append(xmax)
    labels_dict['ymax'].append(ymax)

#print(labels_dict)

#convert dictionary into dataframe
df = pd.DataFrame(labels_dict)
print(df.head(10))

df.to_csv('labels.csv', index=False)