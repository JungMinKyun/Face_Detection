from face_detect import init_face_detector, detect_faces_image, close_face_detector
import cv2


init_face_detector()

## For testing
# example_paths = ['Image/lenna.png', 'Image/multi_people.jpg']
# for i, example_path in enumerate(example_paths):
#     img_bgr = cv2.imread(example_path, cv2.IMREAD_COLOR)
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     bounding_boxes, scores = detect_faces_image(img_rgb)

#     print(f'{i}-th face')
#     print(bounding_boxes)
#     print(scores)

# For Real Application
for img_arr in image_arrays:
    img_rgb = img_arr.convert('RGB')
    bounding_boxes, scores = detect_faces_image(img_rgb)

close_face_detector()
