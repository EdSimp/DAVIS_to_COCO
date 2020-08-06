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
    CATEGORIES = [
        {
            "supercategory": "person",
            "id": 1,
            "name": "person",
        },
    ]

    # 检测框的ID起始值
    START_BOUNDING_BOX_ID = 1
    # 类别列表无必要预先创建，程序中会根据所有图像中包含的ID来创建并更新
    PRE_DEFINE_CATEGORIES = {}

    json_dict = {"images": [],
                 "type": "instances",
                 "annotations": [],
                 "categories": []}

    json_dict["categories"].append(CATEGORIES)

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
        json_annos_name = 'instance_val2017.json'
        annos_save_path = './datasets/coco/annotations'
    elif data_type == 'train':
        test_name = 'ImageSets/train.txt'
        dataset_path = './datasets/coco/train2017'
        annos_path = './datasets/coco/train2017_anno'
        json_annos_name = 'instance_train2017.json'
        annos_save_path = './datasets/coco/annotations'
    else:
        raise Exception

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

    print("starting generate json file for {}!\n" % data_type)
    # Fifth step to generate json
    gen_coco_anno(annos_path, json_annos_name, dataset_path, annos_save_path)
