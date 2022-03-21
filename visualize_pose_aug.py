# from utils import *
import os
from pathlib import Path
import json
import numpy as np
import scipy
import cv2 as cv
from matplotlib import pyplot as plt
from torchvision import transforms

from capture_coord import capture
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation


from argument import get_args
# from backbone import darknet53
from dataset import BOP_Dataset, collate_fn
import transform

def orb_match():
    img1 = cv.imread("./datasets/speed/images/real/img000240real.jpg")
    img2 = cv.imread("./datasets/speed/images/real/img000120real.jpg")
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # screening keypoints
    min_distance = 10000
    max_distance = 0
    for x in matches:
        if x.distance < min_distance: min_distance = x.distance
        if x.distance > max_distance: max_distance = x.distance
    print('MIN DISTANCE: %f' % min_distance)
    print('MAX DISTANCE: %f' % max_distance)
    good_match = []
    for x in matches:
        if x.distance <= max(2 * min_distance, 10):
            good_match.append(x)
    print('MATCHING NUMBER: %d' % len(good_match))
    outimage = cv.drawMatches(img1, kp1, img2, kp2, good_match, outImg=None)

    fig, ax = plt.subplots(dpi=300, figsize=(12, 5))
    ax.imshow(outimage)
    ax.set_title(f"KEYPOINTS MATCHING")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    # extracting and matching feature points
    points1 = []
    points2 = []
    for i in good_match:
        points1.append(list(kp1[i.queryIdx].pt))
        points2.append(list(kp2[i.trainIdx].pt))
    points1 = np.array(points1, dtype="float32")
    points2 = np.array(points2, dtype="float32")

    # calculate transition matrix
    M = cv.getPerspectiveTransform(points1[1:5], points2[1:5])

    # Normalize for Esential Matrix calaculation
    # pts_l_norm = cv.undistortPoints(np.expand_dims(points1, axis=1), cameraMatrix=Camera.K, distCoeffs=1)
    # pts_r_norm = cv.undistortPoints(np.expand_dims(points2, axis=1), cameraMatrix=Camera.K, distCoeffs=1)

    E, mask = cv.findEssentialMat(points1, points2, 3003.41296928, (960., 600.))
    num, R, t, mask = cv.recoverPose(E, points1, points2, np.array([]), np.array([]), 3003.41296928, (960., 600.),
                                     mask)

    # perspective transformation
    processed = cv.warpPerspective(img1, M, (1920, 1200))

    # Display the original and processed image
    fig, axes = plt.subplots(3, 1, dpi=300)
    axes[0].imshow(img1)
    axes[1].imshow(processed)
    axes[2].imshow(img2)
    plt.tight_layout()
    plt.show()
    return M


if __name__ == "__main__":
    # M = orb_match()
    cfg = get_args()

    # device = 'cuda'
    device = cfg['RUNTIME']['RUNNING_DEVICE']

    internal_K = np.array(cfg['INPUT']['INTERNAL_K']).reshape(3, 3)

    train_trans = transform.Compose(
        [
            transform.Resize(
                cfg['INPUT']['INTERNAL_WIDTH'],
                cfg['INPUT']['INTERNAL_HEIGHT'], internal_K),
            transform.RandomShiftScaleRotate(
                cfg['SOLVER']['AUGMENTATION_SHIFT'],
                cfg['SOLVER']['AUGMENTATION_SCALE'],
                cfg['SOLVER']['AUGMENTATION_ROTATION'],
                cfg['INPUT']['INTERNAL_WIDTH'],
                cfg['INPUT']['INTERNAL_HEIGHT'],
                internal_K),
            transform.Normalize(
                cfg['INPUT']['PIXEL_MEAN'],
                cfg['INPUT']['PIXEL_STD']),
            transform.ToTensor(),
        ]
    )

    train_set = BOP_Dataset(
        cfg['DATASETS']['TRAIN'],
        cfg['DATASETS']['MESH_DIR'],
        cfg['DATASETS']['BBOX_FILE'],
        train_trans,
        cfg['SOLVER']['STEPS_PER_EPOCH'] * cfg['SOLVER']['IMS_PER_BATCH'],
        training=False)

    img, target, meta_info = train_set.getitem1(1)


    image_size = [1200, 1920]
    augment = {'SIM2REAL_AUG': False, 'ROT_IMAGE_AUG': False, 'ROT_AUG': False}
    split = "real"
    dataset = SpeedDataset(data_path, image_size, augment, split=split)

    index = []
    img1 = "img000240real.jpg"
    # img1 = "img000187real.jpg"
    index.append(dataset.image_names.index(img1))
    img2 = "img000120real.jpg"
    index.append(dataset.image_names.index(img2))

    # Load image and gt pose
    # 3. Visualize original image + gt
    # for k in range(2):
    #     fig, ax = plt.subplots(dpi=300, figsize=(8, 5))
    #
    #     dataset.visualize(index[k], ax)
    #     # ax.imshow(image.numpy().transpose(1, 2, 0).squeeze(), cmap="gray")
    #
    #     plt.tight_layout()
    #     plt.show()
    #     plt.close()

    img1 = cv.imread('./datasets/speed/images/real/img000120real.jpg')
    img2 = cv.imread('./datasets/speed/images/real/img000240real.jpg')
    # rows, cols, ch = img.shape
    # pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    # pts1 = capture(img1, "120real")
    pts1 = np.array([[840., 673.],
                     [613., 263.],
                     [430., 603.],
                     [620., 1002.],
                     [931., 827.],
                     [676., 134.],
                     [405., 640.],
                     [526., 873.]], dtype=np.float32)
    # pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    # pts2 = capture(img2, "240real")
    pts2 = np.array([[691., 599.],
                     [842., 113.],
                     [438., 226.],
                     [273., 676.],
                     [641., 789.],
                     [987.,  63.],
                     [392., 236.],
                     [303., 502.]], dtype=np.float32)

    # h = cv.getPerspectiveTransform(pts1, pts2)

    # 计算基础矩阵 采用8点法
    f, mask = cv.findFundamentalMat(points1=pts1, points2=pts2, method=cv.FM_8POINT)
    print("Fundamental Matrix:\n", f)
    # [[ 3.47714141e-06  3.43382482e-06 -2.68906780e-03]
    #  [-3.37070507e-06  3.49152427e-06 -1.02168820e-03]
    #  [-1.12382689e-03 -1.35584016e-03  1.00000000e+00]]

    # 计算本质矩阵
    K = dataset.camera
    e, mask = cv.findEssentialMat(points1=pts1, points2=pts2, cameraMatrix=K)
    print("Essential Matrix:\n ", e)
    # [[ 0.07780953  0.19577942 -0.43749193]
    #  [-0.11070182  0.20297539 -0.4777787 ]
    #  [ 0.69311477  0.04386969 -0.01256527]]

    # 计算单应矩阵
    h, mask = cv.findHomography(pts1, pts2)
    h2, mask = cv.findHomography(pts2, pts1)
    print("Homography Matrix:\n", h)
    # [[ 7.29474047e-01 -7.64495290e-01  5.83979988e+02]
    #  [ 7.65547069e-01  7.48138115e-01 -5.55199463e+02]
    #  [-2.07498154e-05  4.65356971e-06  1.00000000e+00]]

    # 从本质矩阵恢复旋转信息和平移信息
    # retval2, R, t, mask = cv.recoverPose(E=e, points1=pts1, points2=pts2, cameraMatrix=K)
    retval2, R, t, mask = cv.decomposeHomographyMat(H=h, K=K)

    # print("Rotation Matrix - R:\n", R)
    # [[ 0.71375914 -0.6778181  -0.1763817 ]
    #  [ 0.69967698  0.70142107  0.1358698 ]
    #  [ 0.03162283 -0.22038853  0.97489943]]
    # print("Translation Matrix - t:\n", t)
    # [[ 0.72693453]
    #  [-0.6607172 ]
    #  [-0.18713356]]
    # print(mask)


    img1_aug = cv.warpPerspective(img1, h, (1920, 1200))
    img2_aug = cv.warpPerspective(img2, h2, (1920, 1200))
    plt.subplot(221), plt.imshow(img1), plt.title('120real')
    plt.subplot(222), plt.imshow(img1_aug), plt.title('120real-aug')
    plt.subplot(223), plt.imshow(img2), plt.title('240real')
    plt.subplot(224), plt.imshow(img2_aug), plt.title('240real-aug')
    plt.show()

    t1 = dataset.loc_labels[index[0]]
    t2 = dataset.loc_labels[index[1]]
    q1 = dataset.quat_labels[index[0]]
    q2 = dataset.quat_labels[index[1]]





    # (fx, fy) = (K[0, 0], K[1, 1])
    # Ainv = np.array([[1.0 / fx, 0.0, -K[0, 2] / fx],
    #                  [0.0, 1.0 / fy, -K[1, 2] / fy],
    #                  [0.0, 0.0, 1.0]], dtype=np.float32)
    q_r = Quaternion(q1).unit.conjugate * Quaternion(q2).unit
    r = Rotation.from_rotvec(q_r.angle * q_r.axis)



    r_pred = Rotation.from_matrix(R[0])

    from evaluate import orientation_score
    err = orientation_score(r_pred.as_quat(), r.as_quat())

    tvec = t2 - t1
    u = np.dot(Rinv, tvec)  # displacement between camera and world coordinate origin, in world coordinates

    # corners of the image, for here hard coded
    pixel_corners = [scipy.array(c, dtype=scipy.float32) for c in
                     [(0 + 0.5, 0 + 0.5, 1), (0 + 0.5, 640 - 0.5, 1), (480 - 0.5, 640 - 0.5, 1),
                      (480 - 0.5, 0 + 0.5, 1)]]
    scene_corners = []
    for c in pixel_corners:
        lhat = scipy.dot(Rinv, scipy.dot(Ainv, c))  # direction of the ray that the corner images, in world coordinates
        s = u[2] / lhat[2]
        # now we have the case that (s*lhat-u)[2]==0,
        # i.e. s is how far along the line of sight that we need
        # to move to get to the Z==0 plane.
        g = s * lhat - u
        scene_corners.append((g[0], g[1]))

    # now we have: 4 pixel_corners (image coordinates), and 4 corresponding scene_coordinates
    # can call cv2.getPerspectiveTransform on them and so on..
