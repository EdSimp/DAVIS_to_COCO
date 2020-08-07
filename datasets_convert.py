'''
transpose Youku to COCO

COCO annotation:
annotation{
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}
images {"license": 4,
            "file_name": "000000397133.jpg",
            "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
            "height": 427,
            "width": 640,
            "date_captured": "2013-11-14 17:02:52",
            "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
            "id": 397133
            },
'''
import os
import shutil
import glob
import numpy as np
import json
from PIL import Image
from pycococreatortools import pycococreatortools
from tqdm import tqdm


def select_video_images_and_move(youku_path, test_name, images_save_path):
    test_path = os.path.join(youku_path, test_name)

    with open(test_path, 'r') as f:
        video_names = [item.strip() for item in f.readlines()]

    for video in video_names:
        src_path = os.path.join(youku_path, 'JPEGImages', video)
        dst_path = os.path.join(images_save_path, video)
        shutil.copytree(src_path, dst_path)


def select_video_annos_and_move(youku_path, test_name, images_save_path):
    test_path = os.path.join(youku_path, test_name)

    with open(test_path, 'r') as f:
        video_names = [item.strip() for item in f.readlines()]

    for video in video_names:
        src_path = os.path.join(youku_path, 'Annotations', video)
        dst_path = os.path.join(images_save_path, video)
        shutil.copytree(src_path, dst_path)


def move_image_data_to_all(youku_path, test_name, dataset_path):
    test_path = os.path.join(youku_path, test_name)
    with open(test_path, 'r') as f:
        video_names = [item.strip() for item in f.readlines()]

    cnt = 0
    for video in video_names:
        video_path = os.path.join(dataset_path, video)
        if not os.path.isdir(video_path):
            continue

        video_image_names = glob.glob(os.path.join(video_path, '*.jpg'))
        for video_image in video_image_names:
            new_name = str(cnt).zfill(7) + '.jpg'

            shutil.copy(video_image, os.path.join(video_path, '../', new_name))
            # os.rename(video_image, os.path.join(video_path, new_name))
            cnt += 1
        shutil.rmtree(video_path)


def move_anno_data_to_all(youku_path, test_name, dataset_path):
    test_path = os.path.join(youku_path, test_name)
    with open(test_path, 'r') as f:
        video_names = [item.strip() for item in f.readlines()]

    cnt = 0
    for video in video_names:
        video_path = os.path.join(dataset_path, video)
        if not os.path.isdir(video_path):
            continue

        video_image_names = glob.glob(os.path.join(video_path, '*.png'))
        for video_image in video_image_names:
            new_name = str(cnt).zfill(7) + '.png'

            shutil.copy(video_image, os.path.join(video_path, '../', new_name))
            # os.rename(video_image, os.path.join(video_path, new_name))
            cnt += 1
        shutil.rmtree(video_path)


def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.' % (filename))


def gen_coco_anno(annos_path, json_annos_name, dataset_path, annos_save_path):
    CATEGORIES = [{"supercategory": "person", "id": 1, "name": "person"},
                  {"supercategory": "vehicle", "id": 2, "name": "bicycle"},
                  {"supercategory": "vehicle", "id": 3, "name": "car"},
                  {"supercategory": "vehicle", "id": 4, "name": "motorcycle"},
                  {"supercategory": "vehicle", "id": 5, "name": "airplane"},
                  {"supercategory": "vehicle", "id": 6, "name": "bus"},
                  {"supercategory": "vehicle", "id": 7, "name": "train"},
                  {"supercategory": "vehicle", "id": 8, "name": "truck"},
                  {"supercategory": "vehicle", "id": 9, "name": "boat"},
                  {"supercategory": "outdoor", "id": 10, "name": "traffic light"},
                  {"supercategory": "outdoor", "id": 11, "name": "fire hydrant"},
                  {"supercategory": "outdoor", "id": 13, "name": "stop sign"},
                  {"supercategory": "outdoor", "id": 14, "name": "parking meter"},
                  {"supercategory": "outdoor", "id": 15, "name": "bench"},
                  {"supercategory": "animal", "id": 16, "name": "bird"},
                  {"supercategory": "animal", "id": 17, "name": "cat"},
                  {"supercategory": "animal", "id": 18, "name": "dog"},
                  {"supercategory": "animal", "id": 19, "name": "horse"},
                  {"supercategory": "animal", "id": 20, "name": "sheep"},
                  {"supercategory": "animal", "id": 21, "name": "cow"},
                  {"supercategory": "animal", "id": 22, "name": "elephant"},
                  {"supercategory": "animal", "id": 23, "name": "bear"},
                  {"supercategory": "animal", "id": 24, "name": "zebra"},
                  {"supercategory": "animal", "id": 25, "name": "giraffe"},
                  {"supercategory": "accessory", "id": 27, "name": "backpack"},
                  {"supercategory": "accessory", "id": 28, "name": "umbrella"},
                  {"supercategory": "accessory", "id": 31, "name": "handbag"},
                  {"supercategory": "accessory", "id": 32, "name": "tie"},
                  {"supercategory": "accessory", "id": 33, "name": "suitcase"},
                  {"supercategory": "sports", "id": 34, "name": "frisbee"},
                  {"supercategory": "sports", "id": 35, "name": "skis"},
                  {"supercategory": "sports", "id": 36, "name": "snowboard"},
                  {"supercategory": "sports", "id": 37, "name": "sports ball"},
                  {"supercategory": "sports", "id": 38, "name": "kite"},
                  {"supercategory": "sports", "id": 39, "name": "baseball bat"},
                  {"supercategory": "sports", "id": 40, "name": "baseball glove"},
                  {"supercategory": "sports", "id": 41, "name": "skateboard"},
                  {"supercategory": "sports", "id": 42, "name": "surfboard"},
                  {"supercategory": "sports", "id": 43, "name": "tennis racket"},
                  {"supercategory": "kitchen", "id": 44, "name": "bottle"},
                  {"supercategory": "kitchen", "id": 46, "name": "wine glass"},
                  {"supercategory": "kitchen", "id": 47, "name": "cup"},
                  {"supercategory": "kitchen", "id": 48, "name": "fork"},
                  {"supercategory": "kitchen", "id": 49, "name": "knife"},
                  {"supercategory": "kitchen", "id": 50, "name": "spoon"},
                  {"supercategory": "kitchen", "id": 51, "name": "bowl"},
                  {"supercategory": "food", "id": 52, "name": "banana"},
                  {"supercategory": "food", "id": 53, "name": "apple"},
                  {"supercategory": "food", "id": 54, "name": "sandwich"},
                  {"supercategory": "food", "id": 55, "name": "orange"},
                  {"supercategory": "food", "id": 56, "name": "broccoli"},
                  {"supercategory": "food", "id": 57, "name": "carrot"},
                  {"supercategory": "food", "id": 58, "name": "hot dog"},
                  {"supercategory": "food", "id": 59, "name": "pizza"},
                  {"supercategory": "food", "id": 60, "name": "donut"},
                  {"supercategory": "food", "id": 61, "name": "cake"},
                  {"supercategory": "furniture", "id": 62, "name": "chair"},
                  {"supercategory": "furniture", "id": 63, "name": "couch"},
                  {"supercategory": "furniture", "id": 64, "name": "potted plant"},
                  {"supercategory": "furniture", "id": 65, "name": "bed"},
                  {"supercategory": "furniture", "id": 67, "name": "dining table"},
                  {"supercategory": "furniture", "id": 70, "name": "toilet"},
                  {"supercategory": "electronic", "id": 72, "name": "tv"},
                  {"supercategory": "electronic", "id": 73, "name": "laptop"},
                  {"supercategory": "electronic", "id": 74, "name": "mouse"},
                  {"supercategory": "electronic", "id": 75, "name": "remote"},
                  {"supercategory": "electronic", "id": 76, "name": "keyboard"},
                  {"supercategory": "electronic", "id": 77, "name": "cell phone"},
                  {"supercategory": "appliance", "id": 78, "name": "microwave"},
                  {"supercategory": "appliance", "id": 79, "name": "oven"},
                  {"supercategory": "appliance", "id": 80, "name": "toaster"},
                  {"supercategory": "appliance", "id": 81, "name": "sink"},
                  {"supercategory": "appliance", "id": 82, "name": "refrigerator"},
                  {"supercategory": "indoor", "id": 84, "name": "book"},
                  {"supercategory": "indoor", "id": 85, "name": "clock"},
                  {"supercategory": "indoor", "id": 86, "name": "vase"},
                  {"supercategory": "indoor", "id": 87, "name": "scissors"},
                  {"supercategory": "indoor", "id": 88, "name": "teddy bear"},
                  {"supercategory": "indoor", "id": 89, "name": "hair drier"},
                  {"supercategory": "indoor", "id": 90, "name": "toothbrush"}]

    # 检测框的ID起始值
    START_BOUNDING_BOX_ID = 1
    # 类别列表无必要预先创建，程序中会根据所有图像中包含的ID来创建并更新
    PRE_DEFINE_CATEGORIES = {}

    json_dict = {"images": [],
                 "type": "instances",
                 "annotations": [],
                 "categories": []}

    json_dict["categories"] = CATEGORIES

    categories = PRE_DEFINE_CATEGORIES
    anno_id = START_BOUNDING_BOX_ID

    annos_list = glob.glob(os.path.join(annos_path, '*.png'))

    pbar = tqdm(total=len(annos_list))
    image_id = 0

    for anno_name in annos_list:
        anno_mask = Image.open(anno_name)
        anno_filename = anno_name.split('/')[-1]

        unique_mask_num = len(np.unique(np.array(anno_mask)))  # count background
        # image_id = get_filename_as_int(
        #     anno_filename.split('.')[0])  # save for json " images" -> id, "annotation" -> image_id
        object_ids = [o_id for o_id in np.unique(anno_mask) if o_id != 0]

        image_filename = anno_filename.split('.')[0] + '.jpg'  # save for json " images" -> file_name
        width = anno_mask.width  # save for json " images" -> width
        height = anno_mask.height  # save for json " images" -> height

        image_path = os.path.join(dataset_path, image_filename)
        image = Image.open(image_path)
        image_info = pycococreatortools.create_image_info(image_id, image_filename, image.size)
        json_dict["images"].append(image_info)

        # for each object in mask
        for object_id in object_ids:
            this_obj_mask = (np.array(anno_mask) == object_id).astype(np.uint8)
            if this_obj_mask.sum() == 0:
                continue
            class_id = 1
            category_info = {'id': class_id, 'is_crowd': 0}

            anno_info = pycococreatortools.create_annotation_info(anno_id, image_id, category_info, this_obj_mask,
                                                                  image.size, tolerance=2)

            json_dict["annotations"].append(anno_info)

            anno_id += 1
        image_id += 1
        pbar.update(1)
    with open(os.path.join(annos_save_path, json_annos_name), "w") as f:
        json.dump(json_dict, f)


if __name__ == '__main__':
    youku_path = '/home/mk/shiqi/video_analyst/datasets/Youku/'
    data_type = 'train'
    data_gen = True

    if data_type == 'val':
        test_name = 'ImageSets/test.txt'
        dataset_path = './datasets/coco/val2017'
        annos_path = './datasets/coco/val2017_anno'
        json_annos_name = 'instances_val2017.json'
        annos_save_path = './datasets/coco/annotations'
    elif data_type == 'train':
        test_name = 'ImageSets/train.txt'
        dataset_path = './datasets/coco/train2017'
        annos_path = './datasets/coco/train2017_anno'
        json_annos_name = 'instances_train2017.json'
        annos_save_path = './datasets/coco/annotations'
    else:
        raise Exception

    if not os.path.exists(annos_save_path):
        os.makedirs(annos_save_path)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    if not os.path.exists(annos_path):
        os.makedirs(annos_path)

    if data_gen is True:
        print("starting copy and move data!\n")

        pbar = tqdm(total=4)

        # First step to select video image
        select_video_images_and_move(youku_path, test_name, dataset_path)
        pbar.update(1)

        # Second step to merge image data
        move_image_data_to_all(youku_path, test_name, dataset_path)
        pbar.update(1)

        # Third step to select annotations
        select_video_annos_and_move(youku_path, test_name, annos_path)
        pbar.update(1)

        # Forth step to merge anno data
        move_anno_data_to_all(youku_path, test_name, annos_path)
        pbar.update(1)

    # print("starting generate json file for {}!\n" % data_type)
    # Fifth step to generate json
    gen_coco_anno(annos_path, json_annos_name, dataset_path, annos_save_path)
