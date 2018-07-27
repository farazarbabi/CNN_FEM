# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 21:01:57 2018

@author: Faraz
"""

import FEM_NN as fem
import time

x_train, x_test, y_train, y_test = fem.get_data()
xy_train = x_train[:,1:]
force_train = x_train[:,0]

# In[4]:
start_time = time.time()
model = fem.build_model2(10)
model.compile(optimizer = 'adam', metrics = ['accuracy',fem.abs_pred], loss = 'mean_absolute_error')
fit = model.fit([xy_train, force_train], y_train, epochs = 50, batch_size = 100, verbose = 2)
print("--- %s fitting (seconds) ---" % (time.time() - start_time))

# In[2]:

model.summary()
# In[2]:

import numpy as np
from numpy import genfromtxt
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy.ma as ma
import griddata
import pylab 

data = genfromtxt('surfaceCSVs\\0-sigmaSurface.csv', delimiter=' ')
sigma=np.zeros(shape=(1,1071))
sigma.fill(1000/50000)
coordinates=data[:,[0,1]].T

start_time = time.time()

y_pred = model.predict([coordinates.T,sigma.T])

print("--- %s seconds ---" % (time.time() - start_time))

y_predScaled=y_pred*50000
# In[2]:
#import Image
xx=coordinates[0,]
yy=coordinates[1,]
zz=y_predScaled
x=xx.ravel()
y=yy.ravel()
z=zz.ravel()
xi = np.linspace(0, 2, 5000)
yi = np.linspace(0, 2, 5000)

binsize = 0.07
grid, bins, binloc = griddata.griddata(x, y, z, binsize=binsize)
zmin    = grid[np.where(np.isnan(grid) == False)].min()
zmax    = grid[np.where(np.isnan(grid) == False)].max()

palette = plt.matplotlib.colors.LinearSegmentedColormap('jet1',plt.cm.datad['jet'],60)
palette.set_under(alpha=0.0)
extent = (x.min(), x.max(), y.min(), y.max())
plt.rcParams["font.family"] = "arial"
plt.rcParams["font.size"] = "16"
plt.figure(figsize=(10,10))
plt.imshow(grid, extent=extent, cmap=palette, origin='lower', vmin=zmin, vmax=3050, aspect='1', interpolation='bilinear')
cbar=plt.colorbar()

plt.title('SigmaXX Predictive Values for (S=1000)')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.savefig('testplot.png')
#Image.open('testplot.png').save('testplot.jpg','JPEG')

    # In[4]:
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.tri as tri

xx=coordinates[0,]
yy=coordinates[1,]
zz=y_predScaled
x=xx.ravel()
y=yy.ravel()
z=zz.ravel()
xi = np.linspace(0, 2, 5000)
yi = np.linspace(0, 2, 5000)
zi = mlab.griddata(x, y, z, xi, yi)
triang = tri.Triangulation(x, y)

zmin    = grid[np.where(np.isnan(grid) == False)].min()
zmax    = grid[np.where(np.isnan(grid) == False)].max()

plt.figure(figsize=(10,10))
plt.tricontour(x, y, z, 20, linewidths=0.4, colors='k')
plt.tricontourf(x, y, z, 20, norm=plt.Normalize(vmax=zmax, vmin=zmin))
#plt.subplots_adjust(hspace=0.5)
plt.colorbar() 
plt.title('SigmaXX Predictive Values for (S=1000)')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.savefig('plot2.png')
# griddata and contour.
#xi = np.linspace(0, 2, 1000)
#yi = np.linspace(0, 2, 1000)
#zi = mlab.griddata(x, y, z, xi, yi)

#plt.contour(xi, yi, zi, 20, interp='linear', linewidths=0.5, colors=('r', 'green', 'blue' ))
#plt.contourf(xi, yi, zi, 20,
             #norm=plt.Normalize(vmax=zmax, vmin=zmin)
plt.show()

