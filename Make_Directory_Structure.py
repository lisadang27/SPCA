#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
from zipfile import ZipFile


# In[ ]:


rootpath = '/homes/picaro/bellt/research/KELT-16b/'


# <br>
# 
# # Create the required file directory structure and move/unzip files as needed
# 
# <br>

# In[ ]:


zips = os.listdir(rootpath)
zips = [zips[i] for i in range(len(zips)) if '.zip' in zips[i]]


# In[ ]:


channels = []
for i in range(len(zips)):
    with ZipFile(rootpath+zips[i], 'r') as zipObj:
        channels.append(zipObj.namelist()[0].split('/')[1])

channels_uni = np.unique(channels)


# In[ ]:


aors = []
for i in range(len(zips)):
    with ZipFile(rootpath+zips[i], 'r') as zipObj:
        aors.extend([zipObj.namelist()[j].split('/')[0] for j in range(len(zipObj.namelist()))])
        
aors = np.unique(aors)
aors = [aor for aor in aors if '.txt' not in aor]


# In[ ]:


os.mkdir(rootpath+'data')
for channel in channels_uni:
    os.mkdir(rootpath+'data/'+channel)

for i in range(len(zips)):
    with ZipFile(rootpath+zips[i], 'r') as zipObj:
        zipObj.extractall(rootpath+'data/'+channels[i])


# In[ ]:


os.mkdir(rootpath+'analysis')
for channel in channels_uni:
    os.mkdir(rootpath+'analysis/'+channel)

os.mkdir(rootpath+'analysis/frameDiagnostics')
for channel in channels_uni:
    os.mkdir(rootpath+'analysis/frameDiagnostics/'+channel)
    
os.mkdir(rootpath+'analysis/photometryComparison')
for channel in channels_uni:
    os.mkdir(rootpath+'analysis/photometryComparison/'+channel)


# In[ ]:


os.mkdir(rootpath+'raw')
for i in range(len(zips)):
    os.rename(rootpath+zips[i], rootpath+'raw/'+zips[i])


# In[ ]:


aorSnip = ''
if np.all([aor[:5]==aors[0][:5] for aor in aors[1:]]):
    aorSnip = aors[0][:5]
elif np.all([aor[:4]==aors[0][:4] for aor in aors[1:]]):
    aorSnip = aors[0][:4]
elif np.all([aor[:3]==aors[0][:3] for aor in aors[1:]]):
    aorSnip = aors[0][:3]
elif np.all([aor[:2]==aors[0][:2] for aor in aors[1:]]):
    aorSnip = aors[0][:2]
else:
    aorSnip = aors[0][:1]
print('Your AOR snippet is', aorSnip)
with open(rootpath+'analysis/aorSnippet.txt', 'w') as f:
    f.write(aorSnip)


# In[ ]:




