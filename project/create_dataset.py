## Simple script to extract frames from a videos
## https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

from cv2 import cv2

# to do:
    # 1 ridimensionare img ---> cv2 o PIL? --> lo fa nella classe direttamente
    # distribuire randomicamente tutte le label tra test.txt e train.txt

img_counter = 0

## se necessario riadatta con il for del main
def getAllFrames(path):
    # Opens the Video file
    cap = cv2.VideoCapture(path)
    i=0

    while(cap.isOpened()):
        ret, frame = cap.read() ## returns a bool (True/False). If frame is read correctly, it will be True. So you can check end of the video by checking this return value.
        if ret == False:
            break
        cv2.imwrite('static/datasets/img/01/dataset'+str(i)+'.jpg',frame)
        i+=1

    cap.release()

    return True


def getFrame(path, n, labels_file, dataset_class):
    global img_counter

    video = cv2.VideoCapture(path)

    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    frames_step = total_frames // n

    ## divide with integral result (discard remainder)

    # print("total frames:        ", total_frames,
    #     "\nframe step:          ", frames_step)

    for i in range(n):
        #here, we set the parameter 1 which is the frame number to the frame (i*frames_step)
        video.set(1,i*frames_step)
        ret,frame = video.read()  
        
        if ret == False:
            break

        ## correggi deve salvare tutte le immagini in una cartella col numero progressivo
        ## tipo empty 1, full_2 ecc.. 
         
        # print(frame.shape)
        path_newImg = 'static/datasets/img/trashbean_'+str(img_counter)+'.jpg'
        label = 'trashbean_'+str(img_counter)+'.jpg'

        labels_file.write(label + ', ' + str(dataset_class))
        labels_file.write("\n")

        cv2.imwrite(path_newImg, frame)
        img_counter = img_counter + 1

    video.release()
    return True

def main():
    path_vid = 'static/datasets/videos/'
    ext = '.mp4'

    # i nomi della classe coincideranno con i nomi dei video di acquisizione contenuti
    # nelle varie cartelle in modo da poter estrarre autonomamente il dataset

    class_dict = {
        "empty": -1,
        "half": 0,
        "full": 1
    }
    # for key in class_dict:
    #     print(key, '-->', class_dict[key])

    source_folders_arr = ["01", "02", "03"]    ## 02 non presente ma usato per test

    # n di frame per video
    frames_per_video = 100

    labels_txt = open('static/datasets/all_labels.txt', 'a')

    for source in (source_folders_arr):       ## scan del numero del dataset
        for key in (class_dict):         ## scan delle classi del dataset
            label = getFrame(path_vid + source + '/' + key + ext, frames_per_video, labels_txt, class_dict[key])            

    labels_txt.close()

main()