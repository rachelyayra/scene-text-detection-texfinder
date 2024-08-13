import cv2
import numpy as np
import pytesseract

def load_east_model(model_path):
    net = cv2.dnn.readNet(model_path)
    return net

def decode_predictions(scores, geometry, min_confidence=0.5):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(numCols):
            if scoresData[x] < min_confidence:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    # print(f'These are the results{rects, confidences}')

    return (rects, confidences)

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(probs)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")


def detect_text(image, net, min_confidence=0.5):
    orig = image.copy()
    (H, W) = image.shape[:2]

    newW, newH = (320, 320)
    rW, rH = W / float(newW), H / float(newH)
    image = cv2.resize(image, (newW, newH))

    blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    (rects, confidences) = decode_predictions(scores, geometry, min_confidence)
    boxes = non_max_suppression(np.array(rects), probs=confidences)



    results = []
    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        roi = orig[startY:endY, startX:endX]
        text = pytesseract.image_to_string(roi, config='--psm 6')
        results.append((startX, startY, endX, endY, text.strip()))

    return results



def group_by_distance(boxes):
    # Calculate the centers of the bounding boxes
    centers = []
    for (x1, y1, x2,y2, text) in boxes:

      # Find the center of the boxes.
      xCenter = (x1 + x2) / 2
      yCenter = (y1 + y2) / 2

      centers.append([xCenter, yCenter])
    # print(f'Centers: {centers}')

    data = np.array(centers)


    k = 2  # Typically min_samples - 1
    clustering = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(data)

    labels = clustering.labels_



    unique_labels = set(labels)
    clusters = {}
    for label in unique_labels:
      matching_boxes = [box for i, box in enumerate(boxes) if labels[i] == label]
      sorted_boxes = sorted(matching_boxes, key=lambda x: (x[1],x[0]))
      clusters[label] = sorted_boxes

    # print(f'Clusters:{clusters}')

    grouped_boxes = []
    for key in clusters :
      merged = merge_boxes(clusters[key])
      grouped_boxes.append(merged)

    return grouped_boxes


def extract_phrases(clusters):
    grouped_boxes = []

    for cluster in clusters:
      box = cluster[0]
      words = cluster[1]
      sentence = " ".join(words)
      grouped_boxes.append([box, sentence])

    return grouped_boxes


def merge_boxes(boxes):
    min_x1, min_y1 = float('inf'), float('inf')
    max_x2, max_y2 = float('-inf'), float('-inf')
    texts = []

    for (x1, y1, x2, y2,text) in boxes:
        min_x1 = min(min_x1, x1)
        min_y1 = min(min_y1, y1)
        max_x2 = max(max_x2, x2)
        max_y2 = max(max_y2, y2)
        texts.append(text)

    return (min_x1, min_y1, max_x2, max_y2),(texts)