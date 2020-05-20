#!/usr/bin/env python
# coding: utf-8

# # Regression Script
# ## Random Forest for Remote Sensing Data
# 
# ### Florian Beyer
# 
# 2020-02-21


#--- Requried Packages
from osgeo import gdal, ogr, gdal_array # I/O image data
import numpy as np # math and array handling
import matplotlib.pyplot as plt # plot figures
import pandas as pd # handling large data as table sheets

from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import AdaBoostRegressor
#from sklearn import svm
#from sklearn.kernel_ridge import KernelRidge
#from sklearn.model_selection import GridSearchCV
#from sklearn.cross_decomposition import PLSRegression
#from sklearn.gaussian_process import GaussianProcessRegressor

from joblib import dump, load # package to save trained models

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()


#--- data input
# define a number of trees that should be used (default = 500)
est = 500

# the remote sensing image you want to classify
img_RS = 'D:\\OwnCloud\\WetScapes\\2020_02_21_Regression_Biomass\\Daten\\stack_all_data_tif.tif'

# training and validation
field = 'D:\\OwnCloud\\WetScapes\\2020_02_21_Regression_Biomass\\Daten\\biomasse_all.shp'
# what is the attributes name of your classes in the shape file (field name of the classes)?
attribute = 'FM_in_Gram'

# save path, predicted image
prediction_map = 'D:\\OwnCloud\\WetScapes\\2020_02_21_Regression_Biomass\\Results\\prediction_map.tif'

# save path, trained model
save_model = 'D:\\OwnCloud\\WetScapes\\2020_02_21_Regression_Biomass\\Results\\RFR.joblib'


#--- Data preparation
# load image data
img_ds = gdal.Open(img_RS, gdal.GA_ReadOnly)

img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
               gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
for b in range(img.shape[2]):
    img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()


#--- Data preparation
# load training data from shape file

#model_dataset = gdal.Open(model_raster_fname)
shape_dataset = ogr.Open(field)
shape_layer = shape_dataset.GetLayer()
mem_drv = gdal.GetDriverByName('MEM')
mem_raster = mem_drv.Create('',img_ds.RasterXSize,img_ds.RasterYSize,1,gdal.GDT_UInt16)
mem_raster.SetProjection(img_ds.GetProjection())
mem_raster.SetGeoTransform(img_ds.GetGeoTransform())
mem_band = mem_raster.GetRasterBand(1)
mem_band.Fill(0)
mem_band.SetNoDataValue(0)

att_ = 'ATTRIBUTE='+attribute
# http://gdal.org/gdal__alg_8h.html#adfe5e5d287d6c184aab03acbfa567cb1
# http://gis.stackexchange.com/questions/31568/gdal-rasterizelayer-doesnt-burn-all-polygons-to-raster
err = gdal.RasterizeLayer(mem_raster, [1], shape_layer, None, None, [1],  [att_,"ALL_TOUCHED=TRUE"])
assert err == gdal.CE_None

roi = mem_raster.ReadAsArray()


#--- Display image and Training data
plt.subplot(121)
plt.imshow(img[:, :, 0], cmap=plt.cm.Greys_r)
plt.title('RS image - first band')

plt.subplot(122)
plt.imshow(roi, cmap=plt.cm.Spectral)
plt.title('Training Image')

plt.show()


#--- Number of training pixels:
n_samples = (roi > 0).sum()
print('We have {n} training samples'.format(n=n_samples))



#---
# Subset the image dataset with the training image = X
# Mask the classes on the training dataset = y
# These will have n_samples rows
X = img[roi > 0, :]
y = roi[roi > 0]

print('Our X matrix is sized: {sz}'.format(sz=X.shape))
print('Our y array is sized: {sz}'.format(sz=y.shape))


#--- train model
RFR = RandomForestRegressor(bootstrap=True,
                             criterion='mse',
                             max_depth=None,
                             max_features='auto',
                             max_leaf_nodes=None,
                             min_impurity_decrease=0.0,
                             min_impurity_split=None,
                             min_samples_leaf=1,
                             min_samples_split=2,
                             min_weight_fraction_leaf=0.0,
                             n_estimators=500,
                             n_jobs=-1, # using all cores
                             oob_score=True,
                             random_state=0,
                             verbose=0,
                             warm_start=False)
RFR.fit(X,y)

#---
# Regression R^2 -> coefficient of determination
RFR.score(X,y)


#--- save the regression model to disk

# Save
dump(RFR, save_model)

# load model:
#RFR = load(save_model)


#--- some outputs
#print(RFR.estimators_)
#print(RFR.feature_importances_)
#print(RFR.n_features_)
#print(RFR.n_outputs_)
#print(RFR.oob_score_)
print(RFR.oob_prediction_)

pred1 = np.array(RFR.oob_prediction_)
obs = np.array(y)
obs = [np.float(i) for i in obs]

print(pred1)
print(obs)

fig,ax = plt.subplots()
ax.scatter(obs,pred1)
plt.show()


print(RFR.feature_importances_)



#--- Predicting the rest of the image

# Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification
new_shape = (img.shape[0] * img.shape[1], img.shape[2])
img_as_array = img[:, :, :].reshape(new_shape)

print('Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))

img_as_array = np.nan_to_num(img_as_array)

prediction_ = RFR.predict(img_as_array)


prediction = prediction_.reshape(img[:, :, 0].shape)
print('Reshaped back to {}'.format(prediction.shape))


#--- saving the pedicted image
cols = img.shape[1]
rows = img.shape[0]

prediction.astype(np.float16)

driver = gdal.GetDriverByName("gtiff")
outdata = driver.Create(prediction_map, cols, rows, 1, gdal.GDT_UInt16)
outdata.SetGeoTransform(img_ds.GetGeoTransform())##sets same geotransform as input
outdata.SetProjection(img_ds.GetProjection())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(prediction)
outdata.FlushCache() ##saves to disk!!
print('Image saved to: {}'.format(prediction_map))

