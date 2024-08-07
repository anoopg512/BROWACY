import argparse
import numpy as np
import cv2
import torch
import torchvision
from models.pfld import PFLDInference, AuxiliaryNet
from mtcnn.detector import detect_faces
from shapely.geometry import Polygon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    checkpoint = torch.load("./checkpoint/snapshot/checkpoint.pth.tar", map_location=device)
    pfld_backbone = PFLDInference().to(device)
    pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
    pfld_backbone.eval()
    pfld_backbone = pfld_backbone.to(device)
    return pfld_backbone

pfld_backbone = load_model()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

def detect_frame(frame):
    img = frame.copy()
    height, width = img.shape[:2]
    bounding_boxes, landmarks = detect_faces(img)

    if not len(bounding_boxes):
        right_area = 0
        left_area = 0
        iou_score = 0
        img_both_eye = frame
        img_corrected = frame
        return right_area, left_area, iou_score, img_both_eye, img_corrected

    for box in bounding_boxes:
        x1, y1, x2, y2 = (box[:4] + 0.5).astype(np.int32)
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        cx = x1 + w // 2
        cy = y1 + h // 2
        size = int(max([w, h]) * 1.1)
        x1 = cx - size // 2
        x2 = x1 + size
        y1 = cy - size // 2
        y2 = y1 + size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        edx1 = max(0, -x1)
        edy1 = max(0, -y1)
        edx2 = max(0, x2 - width)
        edy2 = max(0, y2 - height)

        cropped = img[y1:y2, x1:x2]
        if (edx1 > 0 or edy1 > 0 or edx2 > 0 or edy2 > 0):
            cropped = cv2.copyMakeBorder(cropped, edy1, edy2, edx1, edx2, cv2.BORDER_CONSTANT, 0)

        input_ = cv2.resize(cropped, (112, 112))
        input_ = transform(input_).unsqueeze(0).to(device)
        _, landmarks = pfld_backbone(input_)
        pre_landmark = landmarks[0]
        pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [size, size] - [edx1, edy1]

        right_point = pre_landmark.astype(np.int32)[33:42]
        left_point = pre_landmark.astype(np.int32)[42:51]
        all_points = pre_landmark.astype(np.int32)
        points_left_n = translate_eyebrow2(right_point, left_point, all_points)

        for (x, y) in pre_landmark.astype(np.int32)[33:42]:
            cv2.circle(img, (x1 + x, y1 + y), 1, (0, 255, 0))

        for (x, y) in pre_landmark.astype(np.int32)[42:51]:
            cv2.circle(img, (x1 + x, y1 + y), 1, (0, 255, 0))

        img_both_eye = img.copy()

        for (x, y) in points_left_n.astype(np.int32):
            cv2.circle(img, (x1 + x, y1 + y), 1, (255, 255, 133))

        img_corrected = img.copy()

        new_left_points = convert_points(points_left_n.astype(np.int32), x1, y1)
        right_area = cv2.contourArea(right_point)
        left_area = cv2.contourArea(left_point)
        intersection_area = find_interestion(right_point, points_left_n)
        iou_score = intersection_area / (right_area + left_area)

    return right_area, left_area, iou_score, img_both_eye, img_corrected

def find_interestion(point1, point2):
    poly1 = Polygon(point1)
    poly2 = Polygon(point2)
    intersection_poly = poly1.intersection(poly2)
    intersection_area = intersection_poly.area
    print("Intersection area:", intersection_area)
    return intersection_area

def convert_points(points, x, y):
    new_points = []
    for (a, b) in points:
        x2 = a + x
        y2 = b + y
        new_points.append([x2, y2])
    return np.array(new_points).astype(np.int32)

def translate_eyebrow2(right_point, left_point, all_points):
    dist_rpf_lpf_x = np.abs(all_points[37][0] - all_points[42][0])
    new_left_point = []
    for i in range(len(left_point)):
        if i == 0:
            l_1st_new = np.abs([left_point[i][0] - dist_rpf_lpf_x, left_point[i][1]])
            new_left_point.append(l_1st_new)
        else:
            d1 = dist_rpf_lpf_x
            d2 = (left_point[i][0] - all_points[42][0])
            l_2nd_new = np.abs([left_point[i][0] - (d1 + 2 * d2), left_point[i][1]])
            new_left_point.append(l_2nd_new)
    points = np.array(new_left_point, dtype=np.int32)
    return points

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path', default="./checkpoint/snapshot/checkpoint.pth.tar", type=str)
    args = parser.parse_args()
    return args

def main(args):
    # Add your main function logic here if needed
    pass

if __name__ == "__main__":
    args = parse_args()
    main(args)
