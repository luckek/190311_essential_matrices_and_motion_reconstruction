import sys
import matplotlib.pyplot as plt
from SIFTWrapper import *
from PIL import Image, ExifTags
import numpy as np

# Need this to plot 3d
from mpl_toolkits.mplot3d import axes3d


def triangulate(P0, P1, x1, x2):

    # P0,P1: projection matrices for each of two cameras/images
    # x1,x1: corresponding points in each of two images (If using P that has been scaled by K, then use camera
    # coordinates, otherwise use generalized coordinates)
    A = np.array([[P0[2, 0] * x1[0] - P0[0, 0], P0[2, 1] * x1[0] - P0[0, 1], P0[2, 2] * x1[0] - P0[0, 2], P0[2, 3] * x1[0] - P0[0, 3]],
                  [P0[2, 0] * x1[1] - P0[1, 0], P0[2, 1] * x1[1] - P0[1, 1], P0[2, 2] * x1[1] - P0[1, 2], P0[2, 3] * x1[1] - P0[1, 3]],
                  [P1[2, 0] * x2[0] - P1[0, 0], P1[2, 1] * x2[0] - P1[0, 1], P1[2, 2] * x2[0] - P1[0, 2], P1[2, 3] * x2[0] - P1[0, 3]],
                  [P1[2, 0] * x2[1] - P1[1, 0], P1[2, 1] * x2[1] - P1[1, 1], P1[2, 2] * x2[1] - P1[1, 2], P1[2, 3] * x2[1] - P1[1, 3]]])

    u, s, vt = np.linalg.svd(A)

    return vt[-1]


def intrinsic_cam_mtx(f, cu, cv):

    return np.asarray([[f, 0, cu],
                       [0, f, cv],
                       [0, 0, 1]])


def plot_matches(img1, u1, img2, u2, skip=10):

    h = img1.shape[0]
    w = img1.shape[1]

    fig = plt.figure(figsize=(12, 12))
    I_new = np.zeros((h, 2 * w, 3)).astype(int)
    I_new[:, :w, :] = img1
    I_new[:, w:, :] = img2
    plt.imshow(I_new)
    plt.scatter(u1[::skip, 0], u1[::skip, 1])
    plt.scatter(u2[::skip, 0] + w, u2[::skip, 1])
    [plt.plot([u1[0], u2[0] + w], [u1[1], u2[1]]) for u1, u2 in zip(u1[::skip], u2[::skip])]
    plt.show()


# TODO: switch to piexif
def get_intrinsic_params(img):
    # Get relevant exif data

    exif = {ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS}

    f_length_35 = int(exif['FocalLengthIn35mmFilm'])
    img = np.asarray(img)
    h, w, d = img.shape

    f_length = round(f_length_35 / 36 * w, 4)
    sensor_size = (w // 2, h // 2)

    print("focal length:", f_length)
    print("sensor size:", sensor_size, '\n')

    return f_length, sensor_size


def get_inliers(u1, u2, K):

    # Make homogeneous
    u1 = np.column_stack((u1, np.ones(shape=(u1.shape[0], 1))))
    u2 = np.column_stack((u2, np.ones(shape=(u2.shape[0], 1))))

    K_inv = np.linalg.inv(K)

    x1 = u1 @ K_inv.T
    x2 = u2 @ K_inv.T

    E, inliers = cv2.findEssentialMat(x1[:, :2], x2[:, :2], np.eye(3), method=cv2.RANSAC, threshold=1e-3)

    # Flatten inlier mask for use w/ numy arrays
    inliers = inliers.ravel().astype(bool)

    n_in, R, t, _ = cv2.recoverPose(E, x1[inliers, :2], x2[inliers, :2], cameraMatrix=K)

    # Can use this to plot inliers
    # im_1 = plt.imread(sys.argv[1])
    # im_2 = plt.imread(sys.argv[2])
    # h, w, d = im_1.shape
    # skip = 2
    # fig = plt.figure(figsize=(12, 12))
    # I_new = np.zeros((h, 2 * w, 3)).astype(int)
    # I_new[:, :w, :] = im_1
    # I_new[:, w:, :] = im_2
    # plt.imshow(I_new)
    # plt.scatter(u1[inliers, 0][::skip], u1[inliers, 1][::skip])
    # plt.scatter(u2[inliers, 0][::skip] + w, u2[inliers, 1][::skip])
    # [plt.plot([u1[0], u2[0] + w], [u1[1], u2[1]]) for u1, u2 in zip(u1[inliers][::skip], u2[inliers][::skip])]
    # plt.show()

    return np.hstack((R, t)), x1[inliers, :2], x2[inliers, :2]


def main(argv):

    if len(argv) != 2:
        print("usage: Motion reconstruction: <img1> <img2>")
        sys.exit(1)

    im1 = Image.open(argv[0])

    f, sensor_size = get_intrinsic_params(im1)
    cu, cv = sensor_size

    im_1 = plt.imread(argv[0])
    im_2 = plt.imread(argv[1])

    sw = SIFTWrapper(im_1, im_2)
    u1, u2 = sw.compute_best_matches(0.7)

    # Note that here we are assuming all pictures came from same camera
    # (A safe assumption for now, because we KNOW they came from the same camera)
    K_cam = intrinsic_cam_mtx(f, cu, cv)
    extrinsic_cam, x1_inliers, x2_inliers = get_inliers(u1, u2, K_cam)

    # Note that we are using generalized
    # camera coordinates, so we do not
    # need to multiply by K(For either of P0, P1)
    P_0 = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]])

    P_1 = extrinsic_cam

    point_estimates = []
    for x1_pt, x2_pt in zip(x1_inliers, x2_inliers):

        general_point = triangulate(P_0, P_1, x1_pt, x2_pt)
        general_point /= general_point[3]

        point_estimates.append(general_point[:3])

    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for point in point_estimates:

        # print(point)
        x, y, z = point
        ax.scatter(x, y, z, alpha=0.8, edgecolors='none', s=30)

    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
