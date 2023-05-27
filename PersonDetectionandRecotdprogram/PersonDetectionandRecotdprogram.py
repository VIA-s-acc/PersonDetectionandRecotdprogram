import cv2 
import imutils
import time

def detect_person(frame, person_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    persons = person_cascade.detectMultiScale(gray, scaleFactor=1.11, minNeighbors=5, minSize=(30, 30))
    return persons   

def main(record_msg_int = 0, stop_msg_imt = 0, persons = []):
    person_cascade1 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
    person_cascade2 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
    person_cascade3 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lowerbody.xml')
    person_cascade4 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    person_cascade5 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    person_cascade6 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
    person_cascade7 = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
 
    video_capture = cv2.VideoCapture(0)
    recording = False
    out = None
    
    while True:
        ret, frame = video_capture.read()
        frame = imutils.resize(frame, width=500)
        persons = list(detect_person(frame, person_cascade1)) + list(detect_person(frame, person_cascade2)) + list(detect_person(frame, person_cascade3)) + \
            list(detect_person(frame, person_cascade4)) + list(detect_person(frame, person_cascade5)) + list(detect_person(frame, person_cascade6)) + list(detect_person(frame, person_cascade7))


        if len(persons) > 0 and not recording:
            print("Person detected : record start")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output.avi', fourcc, 7, (frame.shape[1], frame.shape[0]))
            recording = True    

        if recording:
            if len(persons) > 0:
                if record_msg_int == 1:
                    print("Person detected : record continue")
                    record_msg_int = 0;
                stop_msg_imt = 1
                last_person_time = time.time()
                for (x, y, w, h) in persons:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    out.write(frame)
            elif time.time() - last_person_time < 5:          
                    out.write(frame)
            elif time.time() - last_person_time > 5:
                if stop_msg_imt == 1:
                    print("No person : record stopped")
                    stop_msg_imt = 0
                    record_msg_int = 1
               
                

        for (x, y, w, h) in persons:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()