

img_arr = img.imread('G:/MangoQualityGrading/dataset/class 1/IMG_20210703_143211.jpg')
print(img_arr.shape)

# Output: (457, 640, 3)
plt.imshow(img_arr)
(h,w,c) = img_arr.shape
img2D = img_arr.reshape(h*w,c)
print(img2D)
print(img2D.shape)

from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=7) # we shall retain only 7 colors
cluster_labels = kmeans_model.fit_predict(img2D)
print(cluster_labels)



from collections import Counter
labels_count = Counter(cluster_labels)
print(labels_count)

# Sample output
Counter({0: 104575, 6: 49581, 4: 36725, 5: 34004, 2: 26165, 3: 23453, 1: 17977})

print(kmeans_model.cluster_centers_)

rgb_cols = kmeans_model.cluster_centers_.round(0).astype(int)

print(rgb_cols)

img_quant = np.reshape(rgb_cols[cluster_labels],(h,w,c))

plt.imshow(img_quant)

fig, ax = plt.subplots(1,2, figsize=(16,12))
ax[0].imshow(img_arr)
ax[0].set_title('Original Image')
ax[1].imshow(img_quant)
ax[1].set_title('Color Quantized Image')





