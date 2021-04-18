## Simple script to extract frames from a videos
## https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

import cv2

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


def getFrame(path, n):

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

        ## correggi deve salvare tutte le immagini in una cartella col numero progressivo
        ## tipo empty 1, full_2 ecc..  
        cv2.imwrite('static/datasets/img/' + '/trashbean_'+str(i)+'.jpg', frame)

    video.release()
    return True

def main():
    
    path_vid = 'static/datasets/videos/'
    ext = '.mp4'

    class_dict = {
        "empty": -1,
        "half": 0,
        "full": 1
    }

    for key in class_dict:
        print(key, '-->', class_dict[key])


    source_folder_arr = ["01", "02"]    ## 02 non presente ma usato per test
    video_name_arr = ["vuoto", "mezzo", "pieno"]

    n = 3

    for source in (source_folder_arr):       ## scan del numero del dataset
        for video in (video_name_arr):         ## scan delle classi del dataset
            getFrame(path_vid + source + '/' + video + ext, n)

main()