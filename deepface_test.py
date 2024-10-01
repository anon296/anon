from deepface import DeepFace
import numpy as np

#result = DeepFace.verify(
#  img1_path = "data/scraped_char_faces/silence-of-lambs/Clarice Starling/img2.jpg",
#  img2_path = "data/faceframes/silence-of-lambs/silence-of-lambs_scene0/face1.jpg",
#)

starling_ref = np.array(DeepFace.represent(img_path = "data/scraped_char_faces/silence-of-lambs/Clarice Starling/img3.jpg",)[0]['embedding'])
starling1 = np.array(DeepFace.represent(img_path = "data/faceframes/silence-of-lambs/silence-of-lambs_scene1/face2.jpg",)[0]['embedding'])
crawford1 = np.array(DeepFace.represent(img_path = "data/faceframes/silence-of-lambs/silence-of-lambs_scene1/face1.jpg",)[0]['embedding'])
starling2 = np.array(DeepFace.represent(img_path = "data/faceframes/silence-of-lambs/silence-of-lambs_scene1/face5.jpg",)[0]['embedding'])
crawford2 = np.array(DeepFace.represent(img_path = "data/faceframes/silence-of-lambs/silence-of-lambs_scene1/face12.jpg",)[0]['embedding'])
breakpoint()

print(result)
