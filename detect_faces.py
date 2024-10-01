import face_recognition
import os
import matplotlib.pyplot as plt
from dl_utils.misc import check_dir


epname = 'oltl-10-18-10'
keyframes_dir = f'SummScreen/ffmpeg-scenes/{epname}'
facemasks_dir = f'SummScreen/ffmpeg-scenes-facemasks/{epname}'
check_dir(facemasks_dir)
for fn in os.listdir(keyframes_dir):
    image = face_recognition.load_image_file(os.path.join(keyframes_dir, fn))
    face_locations = face_recognition.face_locations(image)
    for top, right, bottom, left in face_locations:
        image[top-5:bottom+5, left-5:right+5] = 0
        plt.imshow(image)
        plt.savefig(os.path.join(facemasks_dir, fn.removesuffix('.jpg')+'-facemasks.jpg'))
        if len(face_locations)>1:
            #plt.imshow(image); plt.show()
            print(os.path.join(facemasks_dir, fn.removesuffix('.jpg')+'-facemasks.jpg'))

