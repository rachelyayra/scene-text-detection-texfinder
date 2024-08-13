import argparse
from text_detection import load_east_model, detect_text, group_by_distance, extract_phrases
import cv2

def main():
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Text Detection Pipeline")
    parser.add_argument('--image', required=True, help="Path to the input image")
    parser.add_argument('--output', default="output_with_boxes.jpg", help="Path to save the output image")
    
    args = parser.parse_args()
    # Load the image
    image = cv2.imread(args.image)

    # Load the EAST model
    east_net = load_east_model(east_model_path)

    # Detect text
    text_detections = detect_text(image, east_net)
    # print(f'These are the text detectors{text_detections}')

    groups = group_by_distance(text_detections)
    # print(f'These are the groups{groups}')

    output = extract_phrases(groups)
    # print(f'This are the results{output}')

    # Draw bounding boxes around detected text
    for ((startX, startY, endX, endY), text) in output:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # print(f"Detected text: '{text}' at coordinates ({startX}, {startY}, {endX}, {endY})")


    # Save the image with bounding boxes (optional)
    output_path = "output_with_boxes.jpg"
    cv2.imwrite(output_path, image)


if __name__ == "__main__":
    main()

