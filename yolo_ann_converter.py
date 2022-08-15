import os
import json
import glob
from tqdm import tqdm
import shutil
from pathlib import Path

IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920
TRAIN_RATIO = 0.9
NUM_VIEWS = 7

def generate_yolo_labels(wt_json, out_dir, out_imgs_dir, ds_split_list):
    json_name = os.path.basename(wt_json).split('.')[0]
    
    frame_multi_view = dict(zip(range(NUM_VIEWS), [[]]*NUM_VIEWS))

    # Load WILDTRACK annotation
    with open(wt_json) as f:
        all_pedestrians = json.load(f)
    
    # Iterate over ann json
    for pedestrian in all_pedestrians:
        for view in pedestrian['views']:
            # Skip if pedestrian is not visible on current view
            if any(map(lambda x: x==-1, [view['xmax'],
                                         view['xmin'],
                                         view['ymax'],
                                         view['ymin']])):
                continue

            # Clip coordinates
            xmax = min(max(view['xmax'], 0), IMAGE_WIDTH)
            xmin = min(max(view['xmin'], 0), IMAGE_WIDTH)
            ymax = min(max(view['ymax'], 0), IMAGE_HEIGHT)
            ymin = min(max(view['ymin'], 0), IMAGE_HEIGHT)

            # Calculate yolo coordinates
            x_center = ((xmax + xmin) / 2) / IMAGE_WIDTH
            y_center = ((ymax + ymin) / 2) / IMAGE_HEIGHT
            width = (xmax - xmin) / IMAGE_WIDTH
            height = (ymax - ymin) / IMAGE_HEIGHT

            ann_string = f'{pedestrian["personID"]} {pedestrian["positionID"]} {x_center} {y_center} {width} {height}'
            frame_multi_view[view['viewNum']].append(ann_string)
    
    # Drop annotation txt files
    for view_num, ann_strings in frame_multi_view.items():
        txt_path = os.path.join(out_dir, f'C{view_num + 1}_{json_name}.txt')
        with open(txt_path, 'w') as f:
            f.write('\n'.join(ann_strings))
        
        ds_split_list.append(os.path.join(out_imgs_dir, f'C{view_num + 1}_{json_name}.png'))
    
    return ds_split_list

def move_images(src_dir, dst_dir):
    dir_name = os.path.basename(src_dir) # get name of the camera
    images = os.listdir(src_dir)

    for image in images:
        src_path = os.path.join(src_dir, image)
        dst_path = os.path.join(dst_dir, f'{dir_name}_{image}')
        shutil.move(src_path, dst_path)

def main():
    input_ann_dir = '/home/djordje/Documents/Projects/mv_detection/Wildtrack_dataset/annotations_positions'
    input_img_dir = '/home/djordje/Documents/Projects/mv_detection/Wildtrack_dataset/Image_subsets'
    out_ann_dir = '/home/djordje/Documents/Projects/mv_detection/Wildtrack_dataset/labels'
    out_imgs_dir = '/home/djordje/Documents/Projects/mv_detection/Wildtrack_dataset/images'

    Path(out_ann_dir).mkdir(exist_ok=True, parents=True)
    Path(out_imgs_dir).mkdir(exist_ok=True, parents=True)

    # Move and rename images
    print('Move images from the original per-camera structure into flat structure.')
    # for cam_dir in os.listdir(input_img_dir):
    #     move_images(os.path.join(input_img_dir, cam_dir), out_imgs_dir)

    # Load annotation json files
    ann_jsons = sorted(glob.glob(os.path.join(input_ann_dir, '*.json')))
    print(f'Total annotation json files: {len(ann_jsons)}')

    # Split to the train and validation datasets
    split_index = int(len(ann_jsons) * TRAIN_RATIO)
    train_jsons = ann_jsons[:split_index]
    val_jsons = ann_jsons[split_index:]

    train_imgs = []
    val_imgs = []

    # Process training dataset
    print('Convert training dataset to the yolo format.')
    for json_path in tqdm(train_jsons):
        generate_yolo_labels(json_path, out_ann_dir, out_imgs_dir, train_imgs)
    
    # Process validation dataset
    print('Convert validation dataset to the yolo format.')
    for json_path in tqdm(val_jsons):
        generate_yolo_labels(json_path, out_ann_dir, out_imgs_dir, val_imgs)
    
    # Save final train and val list of annotation files
    with open('/home/djordje/Documents/Projects/mv_detection/Wildtrack_dataset/train.txt', 'w') as f:
        f.write('\n'.join(train_imgs))
    
    with open('/home/djordje/Documents/Projects/mv_detection/Wildtrack_dataset/val.txt', 'w') as f:
        f.write('\n'.join(val_imgs))

if __name__ == '__main__':
    main()