import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

states = {
    "AN": "ANDAMAN AND NICOBAR ISLANDS",
    "AP": "ANDHRA PRADESH",
    "AR": "ARUNACHAL PRADESH",
    "AS": "ASSAM",
    "BR": "BIHAR",
    "CH": "CHANDIGARH",
    "CG": "CHHATTISGARH",
    "DD": "DAMAN AND DIU",
    "DL": "DELHI",
    "DN": "DADRA AND NAGAR HAVELI",
    "GA": "GOA",
    "GJ": "GUJARAT",
    "HR": "HARYANA",
    "HP": "HIMACHAL PRADESH",
    "JH": "JHARKHAND",
    "JK": "JAMMU AND KASHMIR",
    "KA": "KARNATAKA",
    "KL": "KERALA",
    "LA": "LADAKH",
    "LD": "LAKSHADWEEP",
    "MH": "MAHARASHTRA",
    "ML": "MEGHALAYA",
    "MN": "MANIPUR",
    "MP": "MADHYA PRADESH",
    "MZ": "MIZORAM",
    "NL": "NAGALAND",
    "OD": "ODISHA",
    "PB": "PUNJAB",
    "PY": "PUDUCHERRY",
    "RJ": "RAJASTHAN",
    "SK": "SIKKIM",
    "TN": "TAMIL NADU",
    "TS": "TELANGANA",
    "TR": "TRIPURA",
    "UK": "UTTARAKHAND",
    "UP": "UTTAR PRADESH",
    "WB": "WEST BENGAL"
}

"""READ THE IMAGE , GREYSCALE THE IMAGE AND BLUR THE IMAGE"""

img = cv2.imread('image1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

"""APPLYING FILTER AND FINDING EDGES FOR LOCALIZATION"""

bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise reduction
edged = cv2.Canny(bfilter, 30, 200) #Edge detection
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

"""FINDING COUNTOURS AND APPLYING MASK"""

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

"""USING EASYOCR TO READ TEXT"""

reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)
result

num=result[0][1]

"""RENDERING THE RESULT"""

text = result[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))

print("\n",num)
key=num[0:2]
try:
  print("\nTHIS CAR BELONGS TO",states[key],"\n")
except:
  print("\nSTATE NOT FOUND\n")
