import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])  # converts RBG image to grayscale using very specific factors

I = mpimg.imread('a_image.tif')
# I = plt.imread("a_image.tif")
# I = plt.imread("test_8bit.jpg")

I_gray = np.array(rgb2gray(I))
I_gray2 = I_gray.astype(int)
# I_gray = int(rgb2gray(I))
plt.imshow(I_gray, cmap = plt.get_cmap('gray'))
plt.show()

print(I_gray.ravel)

# print(f)
#
# plt.hist(f)
# plt.show()

# in addition, here is some more


'''
temp=np.asarray(Image.open('map.jpeg'))

x=temp.shape[0]
y=temp.shape[1]*temp.shape[2]
temp.resize((x,y)) # a 2D array
print(temp)

# I = plt.imread("a_image.tif")
# I = plt.imread("MARBLES.tif")
I = plt.imread("barbara_gray.bmp", )
I = np.array(I)

plt.imshow(I)
plt.show()

x=I.shape[0]
y=I.shape[1]*I.shape[2]
I.resize((x,y)) # a 2D array
print(I)

plt.plot(I)
plt.show()





#
# aa= np.array([[1,2,3,4,5],[2,2,2,2,2]])
# aaa= np.array([[5,6,7,8,9],[10,11,12,14,15]])
# a = np.array([aa,aaa])
# print(a)
# print(np.shape(a))
# b = a[0]
# print(b)

print (I)
print("poep")
print(I[0])

'''