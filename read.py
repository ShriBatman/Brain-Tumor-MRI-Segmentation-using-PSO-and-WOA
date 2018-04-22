import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
import cv2
import os
import pso
import WOA
import feature
import enhance
import cPickle
testing = "D:\\images\\"

def getclus(images, noclus, enhanced, imgcnt):
	for it,name in enumerate(images):
		piarr = cv2.imread(enhanced + name ,0)
		mask ,piarr = enhance.skstr(piarr)
		
		r = piarr.shape[0]
		ground = [27.5,72.5,110,142.5]
		col = [100,150,200,255]

		lst = []
		for i in range(r):
			for j in range(r):
				if mask[i][j] == 255:
					lst.append((i,j))

		masked_image = np.zeros((len(lst)),dtype=np.uint8)

		for ii in range(len(lst)):
			i, j = lst[ii]
			masked_image[ii] = piarr[i][j]

		print "clustering..with pso"
		gbest, cluselem , clussize = pso.pso(noclus, masked_image)
		print "done with clustering."

		print "data points are:"
		for i,j in enumerate(clussize):
			print "cluster",i,"contains",j,"elements"
		print "global best \n",gbest

		setc = [0,0,0,0]

		for i in range(noclus):
			val = 256
			for j in range(noclus):
				if abs(gbest[i] - ground[j]) < val:
					val = abs(gbest[i] - ground[j])
					setc[i] = col[j]

		seg_imag = np.zeros((r,r),dtype = np.uint8)
		
		for i in range(noclus):
			for j in range(clussize[i]):
				ii, jj = lst[cluselem[i][j]]
				seg_imag[ii,jj] = setc[i]

		plt.imsave(enhanced[:-1]+"image\\"+str(imgcnt)+"p.jpg",seg_imag,cmap = "gray")
		features = feature.feature_extraction(seg_imag)
		features = np.array([features])
		if it == 0:
			psofeatures = features
		else:
			psofeatures = np.append(psofeatures, features, axis =0 )

		print "clustering..with woa"
		gbest, cluselem , clussize = WOA.woa(noclus, masked_image)
		print "done with clustering."

		print "data points are:"
		for i,j in enumerate(clussize):
			print "cluster",i,"contains",j,"elements"
		print "global best \n",gbest

		for i in range(noclus):
			val = 256
			for j in range(noclus):
				if abs(gbest[i] - ground[j]) < val:
					val = abs(gbest[i] - ground[j])
					setc[i] = col[j]

		seg_imag = np.zeros((r,r),dtype = np.uint8)
		
		for i in range(noclus):
			for j in range(clussize[i]):
				ii, jj = lst[cluselem[i][j]]
				seg_imag[ii,jj] = setc[i]

		plt.imsave(enhanced[:-1]+"image\\"+str(imgcnt)+"w.jpg",seg_imag,cmap ="gray")
		features = np.array([feature.feature_extraction(seg_imag)])
		if it == 0:
			woafeatures = features
		else :
			woafeatures = np.append(woafeatures, features, axis =0 )
		imgcnt += 1
	return woafeatures,psofeatures

enhancedtumour = "D:\\enhancedtumour\\"
enhancednontumour = "D:\\enhancednontumour\\"

# enhance.enhtumour()
# enhance.enhnontumour()
imgcnt = 1
filew = open("featwoa.txt","w")
filep = open("featpso.txt","w")
filet = open("target.txt","w")
images1 = [image for image in os.listdir(enhancedtumour)]
noclus = 4 # for cerebrospinal fluid, white matter, grey matter, abnormality
woat ,psot = getclus(images1, noclus, enhancedtumour, imgcnt)
target = np.array([1 for i in range(len(images1))])

images2 = [image for image in os.listdir(enhancednontumour)]
noclus = 3
imgcnt = len(images1) + 1
woant ,psont = getclus(images2, noclus, enhancednontumour, imgcnt)
targetn = np.array([0 for i in range(len(images2))])
print psont.shape,psot.shape

featpso = np.zeros((len(images1)+len(images2),13))
featwoa = np.zeros((len(images1)+len(images2),13))
for i in range(psot.shape[0]):
	for j in range(13):
		featpso[i][j] = psot[i][j]
		featwoa[i][j] = woat[i][j]

for i in range(psont.shape[0]):
	for j in range(13):
		featpso[i+len(images1)][j] = psont[i][j]
		featwoa[i+len(images1)][j] = woant[i][j]

target = np.append(target,targetn)

for i in range(featwoa.shape[0]):
	for j in range(13):
		filew.write(str(featwoa[i][j]) +'\n')
		filep.write(str(featpso[i][j]) +'\n')
	filet.write(str(target[i])+'\n')
filew.close()
filep.close()
filet.close()
clf_woa = SVC(kernel = "poly", degree = 2)
clf_woa.fit(featwoa, target)

with open('clf_woa.pkl','wb') as woaclf:
	cPickle.dump(clf_woa, woaclf)

# with open('clf_woa.pkl','rb') as woaclf:
# 	clf_woa = cPickle.load(woaclf)

clf_pso = SVC(kernel = "poly", degree = 2)
clf_pso.fit(featpso, target)

with open('clf_pso.pkl','wb') as psoclf:
	cPickle.dump(clf_pso, psoclf)



