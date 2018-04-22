import numpy as np
import cv2

def feature_extraction(image):
	'''
	First order histogram features:
	0. MEAN
	1. VARIANCE
	2. SKEWNESS
	3. KURTOSIS
	4. ENERGY
	5. ENTROPY

	SECOND ORDER C0-OCCURRENCE MATRIX BASED FEATURES
	6. ANGULAR SECOND MOMENT 
	7. CORRELATION 
	8. INERTIA
	9. ABSOLUTE VALUE
	10. INVERSE DIFFERENCE
	11. ENTROPY
	12. MAXIMUM PROBABILITY
	'''
	d = 128
	features = np.zeros((13))
	hist = cv2.calcHist([image], [0], None, [d], [0,255])

	hist /= image.size

	for i in range(d):
		features[0] += i*hist[i]
	mean = features[0]

	for i in range(d):
		features[1] += (i-mean)**2.0 * hist[i]
		features[2] += (i-mean)**3.0 * hist[i]
		features[3] += (i-mean)**4.0 * hist[i]
		features[4] += hist[i]**2.0
		if hist[i] != 0:
			features[5] += hist[i] * np.log2(hist[i])  	
 	
 	var = features[1]
 	if var !=0:
 		features[2] *= var**(-1.5)
 		features[3] *= var**(-2.0) - 3
 	features[5] *= -1

	b = np.zeros((d,d)) 
	ux = np.zeros((d))
	uy = np.zeros((d))
	sdx = np.zeros((d))
	sdy = np.zeros((d))

	for i in range(image.shape[0]-1):
		for j in range(image.shape[1]-1):
			x = int(image[i][j]/2)
			y = int(image[i+1][j+1]/2)
			b[x][y] += 1 
			b[y][x] += 1

	b /= 2.0*(image.shape[0]-1)*(image.shape[1]-1)

	for i in range(d):
		for j in range(d):
			ux[i] += j*b[i][j]
			uy[j] += i*b[i][j]

	for i in range(d):
		for j in range(d):
			sdx[i] += (j-ux[i])**2.0 * b[i][j]
			sdy[j] += (i-uy[j])**2.0 * b[i][j]

	sdx[i] **= 0.5
	sdy[i] **= 0.5

	for i in range(d):
		for j in range(d):
			features[6] += b[i][j]**2.0
			if sdx[i]*sdx[j] != 0:
				features[7] += (i*j*b[i][j] - ux[i]*uy[j])/(sdx[i]*sdx[j])
			features[8] += (i-j)**2.0 * b[i][j]
			features[9] += abs(i-j) * b[i][j]
			features[10] += b[i][j]/(1 + (i-j)**2.0)
			if b[i][j] != 0:
				features[11] += b[i][j] * np.log2(b[i][j])
	features[11] *= -1
	features[12] = max(b.flatten())

	return features








