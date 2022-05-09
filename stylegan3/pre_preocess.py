import glob
import cv2

images = glob.glob("datasets/AGORA_image/*")

# python dataset_tool.py --source=datasets/AGORA_image_resized/ --dest=datasets/AGORA_image_resized_256x256.zip
for i in images:
    image = cv2.imread(i)
    cv2.imwrite(i.replace("AGORA_image", "AGORA_image_resized"), cv2.resize(image[:, 80:560, :], (256, 256)))