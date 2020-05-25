import json
import os
import pandas as pd
import argparse

from convert_coco import convert_format
IoU_thresh = 0.5

dir = './json/'
result_image_dir = './data/'
print_error_image = False

margin_pixel = 15

def calculate_rectangle(image_width, image_height, center_x, center_y, width, height):
    center_x = float(center_x)
    center_y = float(center_y)
    width = float(width)
    height = float(height)
    xmin = int(((center_x * 2 - width) / 2) * image_width)
    xmax = int(((center_x * 2 + width) / 2) * image_width)
    ymin = int(((center_y * 2 - height) / 2) * image_height)
    ymax = int(((center_y * 2 + height) / 2) * image_height)
    xmin = max(0, xmin)
    xmax = max(0, xmax)
    ymin = max(0, ymin)
    ymax = max(0, ymax)
    return xmin, xmax, ymin, ymax

def overlap_check(test_xmin, test_xmax, test_ymin, test_ymax, GT_xmin, GT_xmax, GT_ymin, GT_ymax):
    if test_xmin >= GT_xmin - margin_pixel and test_xmax <= GT_xmax + margin_pixel and test_ymin >= GT_ymin - margin_pixel and test_ymax <= GT_ymax + margin_pixel:
        return True
    else:
        return False

def print_score(scores):
    for score in scores:
            print(score)
            
            FN = scores[score]['Total object'] - scores[score]['TP']
            try:
                Precision = round(scores[score]['TP'] / (scores[score]['TP'] + scores[score]['FP']) * 100, 2)
            except ZeroDivisionError:
                Precision = 0
            try:
                Recall = round(scores[score]['TP'] / (scores[score]['TP'] + FN) * 100, 2)
            except ZeroDivisionError: Recall = 0
            try:
                F1 = round(2 * ((Precision * Recall) / (Precision + Recall)), 2)
            except ZeroDivisionError:
                F1 = 0
            
            print("TP : %d, FP : %d, FN : %d" % (scores[score]['TP'], scores[score]['FP'], FN))
            print("Precision :", Precision, "%, Recall :", Recall, "%, F1-score :", F1, "%")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description = 'SmokeDetection evaluation Script')
    parser.add_argument("--target",help='target json file',type=str)
    args = parser.parse_args()
    return args
def main(args):
    target_name = args.target
   
    target_dir = dir + target_name +'/'

    test_results_filename_list = os.listdir(target_dir)
    out_df = []
    # 테스트 결과 목록 파일 리스트에서 순차적으로 성능 측정 실시
    for test_results_filename in test_results_filename_list:
        test_results = convert_format(target_dir+test_results_filename)

        print(test_results_filename.replace(".json", ""))
        
        # 전체 성능 수치 초기화
        Total_object_num = 0
        TP = 0
        FP = 0
        FN = 0

        # 테스트 결과 목록 파일 하나에서 성능 측정 시작
        for file_num, test_result in enumerate(test_results):
            result_image_name = test_result["frame_id"]
            result_image_path = result_image_dir + str(result_image_name).zfill(6) + '.txt'
            if not os.path.exists(result_image_path):
                continue

            width = 1920
            height = 1080

            GT_path = result_image_path
            GT = open(GT_path, 'r')
            GT_list = GT.readlines()
            GT.close()

            test_count = 0
            GT_object_num = len(GT_list)
            test_object_num = len(test_result['objects'])
            Total_object_num += GT_object_num
            GT_check = [False] * GT_object_num
            Test_check = [False] * test_object_num

            # 테스트 결과 상자 추출
            for test_count, test_object in enumerate(test_result['objects']):
                test_class = test_object['class_id']
                test_center_x = test_object['relative_coordinates']['xmin']
                test_center_y = test_object['relative_coordinates']['ymin']
                test_width = test_object['relative_coordinates']['xmax']
                test_height = test_object['relative_coordinates']['ymax']
                
                test_xmin = test_center_x
                test_ymin = test_center_y
                test_xmax = test_width
                test_ymax = test_height
                
                # GT 상자 추출
                for GT_count, GT_object in enumerate(GT_list):
                    GT_object_list = GT_object.split()
                    GT_class = int(GT_object_list[0])
                    GT_center_x = GT_object_list[1]
                    GT_center_y = GT_object_list[2]
                    GT_width = GT_object_list[3]
                    GT_height = GT_object_list[4]

                    GT_xmin, GT_xmax, GT_ymin, GT_ymax = calculate_rectangle(width, height, GT_center_x, GT_center_y, GT_width, GT_height)

                    # GT랑 테스트 결과 비교
                    x1 = max(GT_xmin, test_xmin)
                    y1 = max(GT_ymin, test_ymin)
                    x2 = min(GT_xmax, test_xmax)
                    y2 = min(GT_ymax, test_ymax)

                    area_intersection = (x2 - x1) * (y2 - y1)
                    area_GT = (GT_xmax - GT_xmin) * (GT_ymax - GT_ymin)
                    area_test = (test_xmax - test_xmin) * (test_ymax - test_ymin)
                    area_union = area_GT + area_test - area_intersection

                    iou = area_intersection / area_union
                    overlap = overlap_check(test_xmin, test_xmax, test_ymin, test_ymax, GT_xmin, GT_xmax, GT_ymin, GT_ymax)
                        
                    if test_class == GT_class and (iou >= IoU_thresh or overlap):
                            GT_check[GT_count] = True
                            Test_check[test_count] = True
                    elif print_error_image:
                        print(result_image_path)
                        print("Test\t: ", test_class, test_xmin, test_xmax, test_ymin, test_ymax)
                        print("GT\t: ", GT_class, GT_xmin, GT_xmax, GT_ymin, GT_ymax)
                        print("IoU\t: ", iou)
                        print("Overlap\t: ", overlap)

            # TP, FP 갯수 체크 후 기록
            current_TP = GT_check.count(True)
            current_FP = Test_check.count(False)
            TP += current_TP
            FP += current_FP
            
            # GT는 전체 갯수를 구해야 하기 때문에 순회하면서 다 검사 후 TP인 것도 체크
            for GT_count, GT_object in enumerate(GT_list):
                GT_object_list = GT_object.split()
                GT_width = GT_object_list[3]
                GT_height = GT_object_list[4]


            print("Process Rate : %.2f%% (%d / %d)\r" % ((file_num + 1) / len(test_results) * 100, file_num + 1, len(test_results)), end='')
        
        print("\n")

        # FN은 전체 갯수 - TP
        FN = Total_object_num - TP

        try:
            Precision = round(TP / (TP + FP) * 100, 2)
        except ZeroDivisionError:
            Precision = 0
        try:
            Recall = round(TP / (TP + FN) * 100, 2)
        except ZeroDivisionError:
            Recall = 0
        try:
            F1 = round(2 * ((Precision * Recall) / (Precision + Recall)), 2)
        except ZeroDivisionError:
            F1 = 0

        out_data = [test_results_filename, TP, FP, FN, Precision, Recall, F1]
        out_df.append(out_data)

        print("Total Result")
        print("TP : %d, FP : %d, FN : %d" % (TP, FP, FN))
        print("Precision :", Precision, "%, Recall :", Recall, "%, F1-score :", F1, "%\n\n")


    df2 = pd.DataFrame(out_df,columns = ['FileName','TP','FP','FN','Precision','Recall','F1-score'])
    df2.to_csv(args.target+' '+'testResult.csv')

if __name__ == "__main__":
    args = parse_args()
    main(args)
  

