## Simple script to extract frames from a videos
## https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

from sys import path
from cv2 import cv2
from zipfile import ZipFile
import os
from os.path import basename

# contatore delle immagini totali del dataset
img_counter = 0

def get_frame(path, new_path_img, n_frames, labels_txt, dst_class):
    """ Estrae frame da un video e scrive su labels_txt (label, classe_appartenenza) """
    global img_counter

    video = cv2.VideoCapture(path)
    tot_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    # frames step
    step = tot_frames // n_frames

    print("total frame %d, frame step %d" %(tot_frames, step))

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

def main():
    """ Estrazione immagini da video e creazione del file 'all_labels.txt' """
    class_dict = {
        "empty": 0,
        "half": 1,
        "full": 2
    }

    source_folders_arr = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]

    path_vid = 'dataset/videos/'
    ext = '.mp4'

    # n di frame per video
    frames_per_video = 250

    all_labels = open('dataset/all_labels.txt', 'a')
    all_labels.truncate(0)

    for curr_folder in (source_folders_arr):
        for key in (class_dict):
            get_frame(path_vid + '/' + curr_folder + '/' + key + ext, 'dataset/images/', frames_per_video, all_labels, class_dict[key])
    all_labels.close()
    

    """ Creazione zip file da caricare su google drive """

    zf = ZipFile('dataset/images.zip', "w")
    zf.write('dataset/all_labels.txt')
    for dirname, subdirs, files in os.walk('dataset/images'):
        zf.write(dirname)
        for filename in files:
            zf.write(os.path.join(dirname, filename))
    zf.close()

main()