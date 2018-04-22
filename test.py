import sklearn.svm as SVC
import numpy as np
import feature
import os
import cPickle
import cv2

tumtestdata = "D:\\tumtestdata\\"
nontumtestdata = "D:\\nontumtestdata\\"

tumtest = open("tumtest.txt","w")
nontumtest = open("nontumtest.txt","w")

with open('clf_woa.pkl','rb') as woaclf:
	clf_woa = cPickle.load(woaclf)

with open('clf_pso.pkl','rb') as psoclf:
	clf_pso = cPickle.load(psoclf)

images1 = [image for image in os.listdir(tumtestdata)] 
images2 = [image for image in os.listdir(nontumtestdata)] 
wposi = len(images1)/2
wneg = len(images2)/2
wfposi = 0
wtposi = 0
wfneg = 0
wtneg = 0

pposi = len(images1)/2
pneg = len(images2)/2
pfposi = 0
ptposi = 0
pfneg = 0
ptneg = 0
imgcnt = 1

print "For tumour images..."
for image in range(pposi):
	img = cv2.imread(tumtestdata + str(imgcnt) +'p.jpg' ,0)
	features = feature.feature_extraction(img)
	for i in range(13):
		tumtest.write(str(features[i])+'\n')
	print images1[image]
	print "from pso:",
	if clf_pso.predict([features]) == 1.0:
		print "present, ",
		ptposi += 1
	else :
		print "not present, ",
		pfposi += 1

	img = cv2.imread(tumtestdata + str(imgcnt) +'w.jpg' ,0)
	features = feature.feature_extraction(img)
	for i in range(13):
		tumtest.write(str(features[i])+'\n')

	print "from woa:",
	if clf_woa.predict([features]) == 1.0:
		print "present "
		wtposi +=1
	else :
		print "not present "
		wfposi += 1
	imgcnt += 1

print "For non-tumour images..."
imgcnt = pposi + 1
for image in range(pneg):
	img = cv2.imread(nontumtestdata + str(imgcnt) +'p.jpg' ,0)
	features = feature.feature_extraction(img)
	for i in range(13):
		nontumtest.write(str(features[i])+'\n')
	print images2[image]
	print "from pso:",
	if clf_pso.predict([feature]) == 1.0:
		print "present, ",
		pfneg += 1
	else :
		print "not present, ",
		ptneg += 1

	img = cv2.imread(nontumtestdata + str(imgcnt) +'w.jpg' ,0)
	features = feature.feature_extraction(img)
	for i in range(13):
		nontumtest.write(str(features[i])+'\n')

	print "from woa:",
	if clf_woa.predict([feature]) == 1.0:
		print "present "
		wfneg += 1
	else :
		print "not present"
		wtneg += 1
	imgcnt += 1

print "\n"
print "from above results..."
print "For pso:"
print "Sensitivity = ",(ptposi+0.0)/(pposi)
print "Specificity = ",(ptneg+0.0)/pneg
print "Accuracy = ",(ptposi + ptneg +0.0) / (pposi + pneg) 

print "\nFor woa:"
print "Sensitivity = ",(wtposi+0.0)/(wposi)
print "Specificity = ",(wtneg+0.0)/wneg
print "Accuracy = ",(wtposi + wtneg +0.0) / (wposi + wneg) 






