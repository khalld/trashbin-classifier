## Simple script to extract frames from a videos
## https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

from cv2 import cv2

# nota: ho sistemato il metodo per fare l'estrazione dei video nelle sottocartelle ma l'indice non si resetta in base al type

img_counter = 0

def getFrame(path, path_newImg, n, labels_file, dataset_class):
    global img_counter

    video = cv2.VideoCapture(path)

    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    frames_step = total_frames // n

    ## divide with integral result (discard remainder)

    print("total frames:        ", total_frames,
        "\nframe step:          ", frames_step)

    for i in range(n):
        #here, we set the parameter 1 which is the frame number to the frame (i*frames_step)
        video.set(1,i*frames_step)
        ret,frame = video.read()  
        
        if ret == False:
            break

        labels_file.write('trashbean_'+str(img_counter)+'.jpg' + ', ' + str(dataset_class))
        labels_file.write("\n")

        cv2.imwrite(path_newImg + 'trashbean_'+str(img_counter)+'.jpg', frame)
        img_counter = img_counter + 1

    video.release()
    return True

def main():

    ## nota: resettare il file all_labels, training e test rispettivamente SEMPRE
    ### ******** get frames ****** ####

    class_dict = {
        "empty": 0,
        "half": 1,
        "full": 2
    }

    source_folders_arr = ["01", "02", "03"]
    source_type_datasets = ["test", "training", "validation"]

    path_vid = 'static/datasets/videos/'
    ext = '.mp4'

    # n di frame per video
    frames_per_video = 100

    for source_d in (source_type_datasets):
        curr_path = 'static/datasets/images/' + source_d + '/labels.txt'
        
        labels_txt = open(curr_path, 'a')
        labels_txt.truncate(0)

        for source in (source_folders_arr):       ## scan del numero del dataset
            for key in (class_dict):         ## scan delle classi del dataset
                label = getFrame(   path_vid + source_d + '/' + source + '/' + key + ext,
                                    'static/datasets/images/' + source_d + '/img/',
                                    frames_per_video,
                                    labels_txt, 
                                    class_dict[key]
                                )            

        labels_txt.close()

main()