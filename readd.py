import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
# import cv2
# import os
# import pso
# import WOA
# import feature
# import enhance
import cPickle
testing = "D:\\images\\"
'''
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
'''
# enhance.enhtumour()
# enhance.enhnontumour()
imgcnt = 1
filep = open("featpso.txt","r")
filew = open("featwoa.txt","r")
filet = open("target.txt","r")
d = 311
featpso = np.zeros((d,13))
featwoa = np.zeros((d,13))
target = np.zeros((d),dtype = int)


fp = filep.readlines()
fw = filew.readlines()
tar = filet.readlines()

for i in range(featwoa.shape[0]):
	for j in range(13):
		featpso[i][j] = fp[i*13+j]
		featwoa[i][j] = fw[i*13+j]
	target[i] = int(tar[i])

filep.close()
filew.close()
filet.close()

print featpso,target
print "training woa..."
clf_woa = SVC(kernel = "poly", degree = 1)
clf_woa.fit(featwoa, target)
print "done"

with open('clf_woa.pkl','wb') as woaclf:
	cPickle.dump(clf_woa, woaclf)
print "saved"
# with open('clf_woa.pkl','rb') as woaclf:
# 	clf_woa = cPickle.load(woaclf)

print "training pso..."
clf_pso = SVC(kernel = "poly", degree = 1)
clf_pso.fit(featpso, target)
print "done"
with open('clf_pso.pkl','wb') as psoclf:
	cPickle.dump(clf_pso, psoclf)

print "saved"



