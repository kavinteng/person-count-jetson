import cv2
import time
import numpy as np
import datetime
import csv
import shutil
import os
import requests
import gdown
from pixellib.torchbackend.instance import instanceSegmentation
from pixellib.instance import instance_segmentation

if os.path.isdir('model') == False:
    os.mkdir('model')

file1 = 'model/pointrend_resnet101.pkl'
if os.path.isfile(file1) == False:
    url = "https://drive.google.com/uc?id=1sEbgtKhVMihMi_HHnamknlW_hIiJpUXe"
    gdown.download(url, file1)

file2 = 'model/pointrend_resnet50.pkl'
if os.path.isfile(file2) == False:
    url = "https://drive.google.com/uc?id=1Nk0V_z1QUaAfdMKNp4FaSZmhHKq3dr22"
    gdown.download(url, file2)

file3 = 'model/mask_rcnn_coco.h5'
if os.path.isfile(file3) == False:
    url = "https://drive.google.com/uc?id=1effTCgaRxzY0GcmgGyJf1TyiGuim_2vT"
    gdown.download(url, file3)

def build_folder_file():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vdo_recode = os.path.join(base_dir, 'record')
    backup_img = os.path.join(base_dir, "backup_file")
    date_img = os.path.join(backup_img, "{}".format(datetime.date.today()))
    try:
        os.mkdir(vdo_recode)
    except:
        pass
    try:
        os.mkdir(backup_img)
    except:
        pass
    try:
        os.mkdir(date_img)
    except:
        pass
    try:
        with open('backup_file/Head-count(not for open).csv') as f:
            pass
    except:
        header = ['File name', 'วัน', 'เวลา', 'จำนวนคนทั้งหมด', 'พนักงาน advice', 'ลูกค้า', 'POST STATUS']
        with open('backup_file/Head-count(not for open).csv', 'w', encoding='UTF-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    try:
        with open('{}/landmark.csv'.format(date_img)) as f:
            pass
    except:
        header = ['Time', 'Xmin', 'Ymin', 'Xmax', 'Ymax']
        with open('{}/landmark.csv'.format(date_img), 'w', encoding='UTF-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    return date_img


def build_csv(data):
    try:
        with open('backup_file/Head-count(not for open).csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write multiple rows
            writer.writerows(data)
    except:
        print('File is open, Process will pause')
    try:
        shutil.copyfile('backup_file/Head-count(not for open).csv', 'result.csv')
    except:
        pass


def build_landmark(date_img, landmark):
    try:
        with open('{}/landmark.csv'.format(date_img), 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            # write multiple rows
            writer.writerows(landmark)
    except:
        print('File is open, Process will pause')

def main(url=None, cap=0, model=None, display_alltime=False, display_out=False, time_ref=10, line_notify=5):
    if model == 'pointrend-resnet101':
        segment_image = instanceSegmentation()
        segment_image.load_model("model/pointrend_resnet101.pkl", confidence=0.2, network_backbone="resnet101")
        target_classes = segment_image.select_target_classes(person=True)
        print('load pointrend101')
    elif model == 'pointrend-resnet50':
        segment_image = instanceSegmentation()
        segment_image.load_model("model/pointrend_resnet50.pkl", confidence=0.2)
        target_classes = segment_image.select_target_classes(person=True)
        print('load pointrend50')
    elif model == 'mask-RCNN':
        segment_image = instance_segmentation()
        segment_image.load_model("model/mask_rcnn_coco.h5", confidence=0.2)
        target_classes = segment_image.select_target_classes(person=True)
        print('load mask-rcnn')
    cap = cv2.VideoCapture(cap)

    check_rec = 0
    line_check = 0
    start = None
    size_img_vdo = (1080, 720)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    while True:
        date_img = build_folder_file()
        output = []
        Date = datetime.datetime.now().strftime("%d/%m/%Y")
        Time = datetime.datetime.now().strftime("%T")

        if start == None:
            start = time.time()
        end = time.time()

        _, frame = cap.read()
        if frame is None:
            break
        frame = cv2.resize(frame, size_img_vdo)
        frame_add_in = frame.copy()
        img_record = frame.copy()

        if end - start > time_ref:
            real_check_out = True
            start = None
        else:
            real_check_out = False

        if real_check_out == True:
            employee = 0
            customer = 0
            result, image_result = segment_image.segmentFrame(frame_add_in, show_bboxes=True,
                                                              segment_target_classes=target_classes,
                                                              extract_segmented_objects=True,
                                                              save_extracted_objects=False)

            if model == 'mask-RCNN':
                model_count = result['rois']
            else:
                model_count = result['boxes']
            if len(model_count) != 0:
                for i in range(len(model_count)):
                    output_landmark = []
                    xmin = int(model_count[i][0])
                    ymin = int(model_count[i][1])
                    xmax = int(model_count[i][2])
                    ymax = int(model_count[i][3])
                    # conf = result['scores'][i]
                    # obj_name = result['class_names'][i]
                    extracted = result['extracted_objects'][i]

                    x = xmax - xmin
                    y = ymax - ymin
                    dis = (x * y) / 100

                    xmin_new = xmin
                    ymin_new = ymin
                    xmax_new = xmax
                    ymax_new = int(ymin + (y / 2))
                    if ymax_new > 360:
                        ymax_new = 360

                    try:
                        # shirt = image[ymin_new:ymax_new, xmin_new:xmax_new]
                        lower1 = np.array([167, 111, 60])
                        upper1 = np.array([170, 122, 101])
                        mask1 = cv2.inRange(extracted, lower1, upper1)

                        lower2_1 = np.array([44, 29, 28])
                        upper2_1 = np.array([47, 35, 28])
                        mask2_1 = cv2.inRange(extracted, lower2_1, upper2_1)

                        result_final2_1 = cv2.bitwise_or(mask2_1, mask1)

                        gray_thresh = cv2.adaptiveThreshold(result_final2_1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                            cv2.THRESH_BINARY_INV, 11, 1)
                        kernel = np.ones((3, 3), np.uint8)
                        closing = cv2.morphologyEx(gray_thresh, cv2.MORPH_CLOSE, kernel, iterations=5)
                        contours, hierachy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours == () or contours == []:
                            color = (255, 0, 0)
                        else:
                            array_area = []
                            for cnt in contours:
                                area = cv2.contourArea(cnt)
                                array_area.append(area)

                            out_sum = sum(array_area)
                            thresh = 200
                            if out_sum > thresh:
                                color = (0, 0, 255)
                            else:
                                color = (255, 0, 0)
                    except:
                        color = (255, 0, 0)

                    output_landmark.append([Time, xmin, ymin, xmax, ymax])
                    build_landmark(date_img, output_landmark)

                    if color == (0, 0, 255):
                        employee += 1
                    elif color == (255, 0, 0):
                        customer += 1

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    # cv2.rectangle(frame, (xmin_new, ymin_new), (xmax_new, ymax_new), color, 2)
                    # cv2.putText(frame, 'Head: {:.2f}'.format(acc * 100), (xmin, ymin - 5),
                    #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                    cv2.rectangle(frame, (0, 0), (200, 50), (255, 255, 255), -1)
                    cv2.putText(frame, 'Head Count:{}'.format(employee + customer), (10, 30),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (0, 0, 0), 2)

                    # cv2.imshow('asd',frame)
                    # cv2.imshow('zxc',extracted)
                    # cv2.imshow('dsa',closing)
                    # cv2.waitKey(0)

            line_check += 1
            file_name = Time.replace(':', '-')
            # post json ----------------------------------------------

            count_all_json = employee + customer
            file_json = file_name + '.jpg'
            dd, mm, yyyy = Date.split('/')
            date_json = f'{yyyy}-{mm}-{dd}'
            time_json = date_json + f' {Time}'

            text_for_post = {"img_name": file_json, "img_date": date_json, "img_time": time_json,
                             "people_total": count_all_json, "people_advice": employee,
                             "people_other": customer}

            text = {"Status_post": 'Yes'}

            status_post = request_post(url, text_for_post)
            if status_post == 0:
                text['Status_post'] = 'No'
                print(text_for_post, text)
            elif status_post == 2:
                text['Status_post'] = 'empty url'
                print(text_for_post, text)

            # --------------------------------------------------------
            status_post_csv = text['Status_post']
            output.append([file_json, Date, Time, count_all_json, employee, customer, status_post_csv])

            img_file = date_img + '/' + file_name + '.jpg'
            cv2.imwrite(img_file, frame)
            if display_out:
                cv2.imshow('frame{}'.format(Time), frame)
            build_csv(output)
            if line_check == line_notify and line_notify != False:
                line_check = 0
                try:
                    line_pic('{} {}'.format(Date, Time), img_file)
                except:
                    print('No internet connection')

        if check_rec == 0:
            h, m, s = Time.split(':')
            time_record = '-' + str(h) + '-' + str(m) + '-' + str(s)
            file_record = 'record/' + str(datetime.date.today()) + time_record + '.mp4'
            rec = cv2.VideoWriter(file_record, fourcc, 25, size_img_vdo)
            check_rec = 1
        if check_rec == 1:
            rec.write(img_record)

        if display_alltime == True:
            cv2.imshow('frame', frame)
        k = cv2.waitKey(25)
        if k == ord('q'):
            break
    rec.release()
    cap.release()
    cv2.destroyAllWindows()

def line_pic(message, path_file):
    LINE_ACCESS_TOKEN = "US4BqFvpMcMoTDB4ea9l4bXeGdAA4quMkCzoIWy1Vrb"
    URL_LINE = "https://notify-api.line.me/api/notify"
    file_img = {'imageFile': open(path_file, 'rb')}
    msg = ({'message': message})
    LINE_HEADERS = {"Authorization": "Bearer " + LINE_ACCESS_TOKEN}
    session = requests.Session()
    session_post = session.post(URL_LINE, headers=LINE_HEADERS, files=file_img, data=msg)
    print(session_post.text)


def request_post(url, text):
    if url == None:
        status_post = 2
    else:
        response = requests.post(url, json=text)
        print('------posting------')
        if response.ok:
            print("Upload completed successfully!")
            status_post = 1
            print(response.json())

        else:
            print("Fall upload!")
            response.status_code
            status_post = 0

    return status_post