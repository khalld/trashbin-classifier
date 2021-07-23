## Simple script to extract frames from a videos
## https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

from sys import path
from cv2 import cv2

img_counter = 0

def get_frame(path, new_path_img, n_frames, labels_txt, dst_class):
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

        labels_txt.write('trashbean_%d.jpg, %d\n' %(img_counter, dst_class))
        cv2.imwrite(new_path_img + 'trashbean_' + str(img_counter) + '.jpg', frame )
        img_counter = img_counter + 1

    video.release()
    return True

def main():

    class_dict = {
        "empty": 0,
        "half": 1,
        "full": 2
    }

    source_folders_arr = ["01", "02", "03"]
    source_type_datasets = ["test", "training", "validation"]

    path_vid = 'dataset/videos/'
    ext = '.mp4'

    # n di frame per video
    frames_per_video = 300

    all_labels = open('dataset/all_labels.txt', 'a')
    all_labels.truncate(0)

    for source_dst in (source_type_datasets):
        for curr_folder in (source_folders_arr):
            for key in (class_dict):
                get_frame(path_vid + source_dst + '/' + curr_folder + '/' + key + ext, 'dataset/images/', frames_per_video, all_labels, class_dict[key])

    all_labels.close()

main()