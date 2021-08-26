## Simple script to extract frames from a videos
## https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

from sys import path
from cv2 import cv2
from zipfile import ZipFile
import os
from os.path import basename

# counter of the total number of extracted images
img_counter = 0

def get_frame(path, new_path_img, n_frames, labels_txt, dst_class):
    """ Extract frames from vide a write on `labels_txt` : (label, class) """
    global img_counter

    video = cv2.VideoCapture(path)
    tot_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    # frames step
    step = tot_frames // n_frames

    print('On %s: total frame %d, frame step %d' %(path, tot_frames, step))

    for i in range(n_frames):
        video.set(1, i*step)
        ret, frame = video.read()

        if ret == False:
            break

        labels_txt.write('trashbin_%d.jpg, %d\n' %(img_counter, dst_class))
        cv2.imwrite(new_path_img + 'trashbin_' + str(img_counter) + '.jpg', frame )
        img_counter = img_counter + 1

    video.release()
    return True

if __name__ == "__main__":   
    """ Extract images from videos and create `all_labels.txt` """
    class_dict = {
        "empty": 0,
        "half": 1,
        "full": 2
    }

    source_folders_arr = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']

    path_vid = 'dataset/videos/'
    ext = '.mp4'

    # number of frames for video
    frames_per_video = 400

    print('Extracting frames..')

    all_labels = open('dataset/all_labels.txt', 'a')
    all_labels.truncate(0)

    for curr_folder in (source_folders_arr):
        for key in (class_dict):
            get_frame(path_vid + curr_folder + '/' + key + ext, 'dataset/images/', frames_per_video, all_labels, class_dict[key])
    all_labels.close()

    print('All frames are extracted')

    print('Creating zip file..')

    # create zip file to upload on google drive faster
    zf = ZipFile('dataset/images.zip', "w")
    zf.write('dataset/all_labels.txt')
    for dirname, subdirs, files in os.walk('dataset/images'):
        zf.write(dirname)
        for filename in files:
            zf.write(os.path.join(dirname, filename))
    zf.close()

    print('Zip file created successfully')