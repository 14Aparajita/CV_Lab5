```python
import numpy as np
import cv2
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist

def preprocess_image(img):
    img = cv2.resize(img, (100, 100))
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return thresh

def get_edge_points(img, max_points=100):
    edges = cv2.Canny(img, 50, 150)
    points = np.column_stack(np.where(edges > 0))
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
    return points.astype(np.float32)

def shape_context(points, nbins_r=5, nbins_theta=12):
    r_inner = 0.125
    r_outer = 2.0
    n = len(points)

    dists = np.sqrt(((points[:, None, :] - points[None, :, :])**2).sum(-1))
    mean_dist = dists.mean()
    dists /= mean_dist

    angles = np.arctan2(
        points[:, None, 0] - points[None, :, 0],
        points[:, None, 1] - points[None, :, 1]
    )

    descriptors = []

    for i in range(n):
        hist = np.zeros((nbins_r, nbins_theta))
        for j in range(n):
            if i == j:
                continue

            r = dists[i, j]
            theta = angles[i, j]

            if r < r_inner or r > r_outer:
                continue

            r_bin = int((np.log(r / r_inner) / np.log(r_outer / r_inner)) * nbins_r)
            theta_bin = int(((theta + np.pi) / (2 * np.pi)) * nbins_theta)

            r_bin = min(r_bin, nbins_r - 1)
            theta_bin = min(theta_bin, nbins_theta - 1)

            hist[r_bin, theta_bin] += 1

        descriptors.append(hist.flatten())

    return np.array(descriptors).mean(axis=0)

def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X = np.concatenate([X_train[:1000], X_test[:200]])
    y = np.concatenate([y_train[:1000], y_test[:200]])

    features = []
    labels = []

    for i in range(len(X)):
        img = preprocess_image(X[i])
        pts = get_edge_points(img)
        if len(pts) < 5:
            continue
        desc = shape_context(pts)
        features.append(desc)
        labels.append(y[i])

    features = np.array(features)
    labels = np.array(labels)

    X_tr, X_te, y_tr, y_te = train_test_split(features, labels, test_size=0.2)

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    print("Accuracy:", accuracy_score(y_te, y_pred))

if __name__ == "__main__":
    main()
