import os
import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from tqdm import tqdm
from types import MethodType

# Helper function to encode image
def encode_image(img):
    res = resnet(torch.Tensor(img))
    return res

# Custom detect_box method for MTCNN
def detect_box_custom(self, img, save_path=None):
    # Detect faces
    batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)

    # Select faces
    if not self.keep_all:
        batch_boxes, batch_probs, batch_points = self.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=self.selection_method
        )

    # Extract faces
    faces = self.extract(img, batch_boxes, save_path)
    return batch_boxes, faces

# Load pre-trained models
resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(
    image_size=224, keep_all=True, thresholds=[0.4, 0.5, 0.5], min_face_size=60
)
mtcnn.detect_box_custom = MethodType(detect_box_custom, mtcnn)

# Get encoded features for all saved images
saved_directory = "./saved/"
all_people_faces = {}

for file in os.listdir(saved_directory):
    if file.endswith(".jpg"):
        person_name = os.path.splitext(file)[0]
        img = cv2.imread(os.path.join(saved_directory, file))
        cropped = mtcnn(img)
        if cropped is not None:
            encoding = encode_image(cropped)[0, :]
            all_people_faces[person_name] = encoding

# Function to detect and recognize faces from a video stream
def detect_faces(cam=0, threshold=0.7):
    video_capture = cv2.VideoCapture(cam)

    while video_capture.grab():
        _, frame = video_capture.retrieve()
        batch_boxes, cropped_images = mtcnn.detect_box_custom(frame)

        if cropped_images is not None:
            for box, cropped in zip(batch_boxes, cropped_images):
                x, y, x2, y2 = [int(coord) for coord in box]
                img_embedding = encode_image(cropped.unsqueeze(0))
                detection_scores = {}

                # Compare embeddings with all saved faces
                for name, encoding in all_people_faces.items():
                    detection_scores[name] = (encoding - img_embedding).norm().item()

                # Find the person with the minimum distance
                min_name = min(detection_scores, key=detection_scores.get)

                # Assign 'Undetected' if the minimum distance exceeds the threshold
                if detection_scores[min_name] >= threshold:
                    min_name = 'Undetected'

                # Draw bounding box and display person's name
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame, min_name, (x + 5, y + 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1
                )

        # Display the frame with bounding boxes and names
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    detect_faces(0)
