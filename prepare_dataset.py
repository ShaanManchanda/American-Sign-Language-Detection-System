import copy
import cv2 as cv
import mediapipe as mp
from app_files import calc_landmark_list, draw_landmarks, get_args, pre_process_landmark,logging_csv

def main():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode#Determines if the detection is done on static images
    min_detection_confidence = args.min_detection_confidence#Determines the minimum confidence value for detection
    min_tracking_confidence = args.min_tracking_confidence#Determines the minimum confidence value for tracking

    cap = cv.VideoCapture(cap_device)#specify which camera to use
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    mode = 1
    number = -1
    #main loop
    while True:
        #handling key press
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        
        if 48 <= key <= 57:  # 0 ~ 9
            number = key - 48
            
        #capture frame
        ret, image = cap.read()

        if not ret:
            break

        #image processing
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)        #creates a copy of the image
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False #makes image read only
        results = hands.process(image)
        image.flags.writeable = True
        # cv.imshow("results",results)
        # print(type(results))
        # print(results)

        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks :                               
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)#get the landmark points
                pre_processed_landmark_list = pre_process_landmark(landmark_list)#normalize the landmark points
                logging_csv(number, mode, pre_processed_landmark_list)#save the landmark points to a csv file
                # logging_csv(number, mode, landmark_list)
                debug_image = draw_landmarks(debug_image, landmark_list)##draw landmarks on the image
                info_text="Press key 0-9"
                cv.putText(debug_image, info_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1.0, (196, 161, 33), 1, cv.LINE_AA)
        cv.imshow('Dataset Preparation', debug_image)

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()

