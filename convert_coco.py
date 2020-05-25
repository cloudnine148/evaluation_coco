import json
import copy
import os
from collections import OrderedDict

def cal_coordinate(bbox):
        calc_bbox = []
        for i in range(0,4):
            if bbox[i] < 0:
                bbox[i] = 0.0
        
        bbox[0] = round(bbox[0],4)
        calc_bbox.append(bbox[0])
        bbox[1] = round(bbox[1],4)
        calc_bbox.append(bbox[1])
        bbox[2] = round(bbox[0] + bbox[2],4)
        calc_bbox.append(bbox[2])
        bbox[3] = round(bbox[1] + bbox[3],4)
        calc_bbox.append(bbox[3])
    
        return calc_bbox

def convert_format(target_dir):
    src_list = open(target_dir)
    src_json = json.load(src_list)
    src_list.close()
    
    dst_annos = []
    dst_anno = {}
    
    init_image_id = 0
    cur_image_id = 0
        
    bboxes= list()
    bbox_list = list()
    bbox = list() 
    
   
    
    aFlag = True 
    init_image_id = 0
    cur_image_id =  0
    
    for num,result in enumerate(src_json):
        if aFlag == True:
            init_image_id = result['image_id']
            cur_image_id = result['image_id']
            aFlag = False
    
        cur_image_id = result['image_id']
        confidence_score = result['score']
        if confidence_score < 0.25:
            continue
        if init_image_id == cur_image_id:
            #dst_anno['frame_id'] = cur_image_id
            #dst_annos.append(copy.deepcopy(dst_anno))
            bbox.clear()
            bbox = result['bbox']
            bbox = cal_coordinate(bbox)
            bbox_info = {
                'class_id':0,
                'name' :'smoke',
                'relative_coordinates':{
                    'xmin' : bbox[0],
                    'ymin' : bbox[1],
                    'xmax': bbox[2],
                    'ymax' : bbox[3]
                },
                'confidence': confidence_score
            }
            bbox_list.append(bbox_info)
        else:
            #dst_anno['frame_id'] = cur_image_id - 1
            dst_anno['frame_id'] = init_image_id
            dst_anno['objects'] = bbox_list
            dst_annos.append(copy.deepcopy(dst_anno))
    
            bbox_list.clear()
            init_image_id = cur_image_id
    
            bbox.clear()
            #bbox_info.clear()
            bbox = result['bbox']
            bbox = cal_coordinate(bbox)
            bbox_info = {
                'class_id':0,
                'name' :'smoke',
                'relative_coordinates':{
                    'xmin' : bbox[0],
                    'ymin' : bbox[1],
                    'xmax': bbox[2],
                    'ymax' : bbox[3]
                },
                'confidence': confidence_score
            }
            bbox_list.append(bbox_info)
    
    return dst_annos
