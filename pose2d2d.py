import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from numpy.lib.twodim_base import mask_indices


def find_feature_matches(img_1, img_2):
    orb = cv.ORB_create()

    kp1 = orb.detect(img_1)
    kp2 = orb.detect(img_2)

    kp1, des1 = orb.compute(img_1, kp1)
    kp2, des2 = orb.compute(img_2, kp2)

    bf = cv.BFMatcher(cv.NORM_HAMMING)

    matches = bf.match(des1, des2)

    min_distance = matches[0].distance
    max_distance = matches[0].distance

    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance

    print("Max dist:", max_distance)
    print("Min dist:", min_distance)

    good_match = []

    for x in matches:
        if x.distance <= max(2 * min_distance, 30.0):
            good_match.append(x)

    match_image = cv.drawMatches(img_1, kp1, img_2, kp2, good_match, outImg=None)

    fig, ax = plt.subplots(dpi=300, figsize=(12, 5))
    ax.imshow(match_image)
    ax.set_title("KEYPOINTS MATCHING")
    plt.tight_layout()
    plt.show()

    return kp1, kp2, good_match


def poes_estimation_2d2d(keypoint_1, keypoint_2, matches):
    class Camera:
        fx = 0.0176  # focal length[m]
        fy = 0.0176  # focal length[m]
        nu = 1920  # number of horizontal[pixels]
        nv = 1200  # number of vertical[pixels]
        ppx = 5.86e-6  # horizontal pixel pitch[m / pixel]
        ppy = ppx  # vertical pixel pitch[m / pixel]
        fpx = fx / ppx  # horizontal focal length[pixels]
        fpy = fy / ppy  # vertical focal length[pixels]
        k = [[fpx, 0, nu / 2],
             [0, fpy, nv / 2],
             [0, 0, 1]]
        K = np.array(k)
    # k = [[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1]]
    k = Camera.K
    print("Intrinsic Matrix:\n", k)

    # print(keypoint_1)
    # print("DescriptorMatcher:\n", matches)

    good = []
    pts2 = []
    pts1 = []
    for i in range(int(len(matches))):
        pts1.append(keypoint_1[matches[i].queryIdx].pt)
        pts2.append(keypoint_2[matches[i].trainIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)



    # 计算基础矩阵 采用8点法
    f, mask = cv.findFundamentalMat(points1=pts1, points2=pts2, method=cv.FM_8POINT)
    print("Fundamental Matrix:\n", f)

    # 计算本质矩阵
    e, mask = cv.findEssentialMat(points1=pts1, points2=pts2, cameraMatrix=k)
    print("Essential Matrix:\n ", e)

    # 计算单应矩阵
    h, mask = cv.findHomography(pts1, pts2)
    print("Homography Matrix:\n", h)
    # [[ 6.81194143e-01 -7.64758200e-01  5.95875886e+02]
    #  [ 7.13611840e-01  6.88254273e-01 -5.05604246e+02]
    #  [-2.71959335e-05 -5.52233475e-05  1.00000000e+00]]


    # 从本质矩阵恢复旋转信息和平移信息
    retval2, R, t, mask = cv.recoverPose(E=e, points1=pts1, points2=pts2, cameraMatrix=k)
    print("旋转矩阵R:", R)
    # [[0.70057454 -0.71238835 -0.04120864]
    #  [0.71312455  0.70102093  0.00479928]
    #  [0.02546916 -0.03274914  0.99913904]]
    print("平移矩阵t:", t)
    # [[0.05934216]
    #  [0.08184525]
    #  [-0.99487681]]
    # print(mask)
    return h



if __name__ == "__main__":
    # img_1 = cv.imread("1.png")
    # img_2 = cv.imread("2.png")
    img_1 = cv.imread("data/SwissCube_1.0/training/seq_000000/000000/rgb/000001.jpg")
    img_2 = cv.imread("data/SwissCube_1.0/training/seq_000000/000000/rgb/000009.jpg")

    # 图像匹配
    keypoint_1, keypoint_2, matches = find_feature_matches(img_1, img_2)
    print("Total Matches:\n", len(matches))

    # 预测位姿
    # R, t = poes_estimation_2d2d(keypoint_1, keypoint_2, matches)
    h = poes_estimation_2d2d(keypoint_1, keypoint_2, matches)

    # perspective transformation
    processed = cv.warpPerspective(img_1, h, (1024, 1024))

    # Display the original and processed image
    fig, axes = plt.subplots(2, 2, dpi=300)
    axes[0, 0].imshow(img_1)
    axes[0, 1].imshow(processed)
    axes[1, 0].imshow(img_2)
    plt.tight_layout()
    plt.show()
