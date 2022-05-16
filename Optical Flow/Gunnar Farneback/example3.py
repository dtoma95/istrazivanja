import cv2
import numpy as np
import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow

w,h = 256,256
gray_1 = np.zeros((w,h),dtype = np.uint8)
gray_2 = np.zeros((w,h),dtype = np.uint8)

for i in range(256):
    for j in range(256):
        gray_1[i,j] = i//2+j//2
        gray_2[i,j] = i//2+j//2

dx = 0
dy = 2
mask = np.ones((30,30),dtype = np.uint8)*200  
gray_1[20:50,20:50] = mask
gray_2[20+dy:50+dy, 20+dx:50+dx] = mask 
diff = cv2.subtract(gray_2, gray_1)

cv2.imwrite("squareFrame1.png", gray_1)
cv2.imwrite("squareFrame2.png", gray_2)
#cv2_imshow(gray_1)
#cv2_imshow(gray_2)
#cv2_imshow(diff)

flow = cv2.calcOpticalFlowFarneback(gray_1, gray_2, None, 0.5, 3, 39,  10,  11,  1.5,cv2.OPTFLOW_FARNEBACK_GAUSSIAN) 

#remove outliers
index = (np.abs(flow) > 100)
flow[index] = 0
index = (np.abs(flow)) < 0.001
flow[index] = 0
index_interesting = np.abs(flow) > 0.1

#flow along y axis - v
plt.imshow(flow[:,:,1], cmap = 'gray')

flow_hist = flow[index_interesting]
plt.hist(flow_hist.reshape(-1,1), bins = 50)

hsv = np.zeros( (256,256,3), dtype = np.uint8)
hsv[...,1] = 255
mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
plt.imshow(rgb)
plt.savefig("rgb.png")