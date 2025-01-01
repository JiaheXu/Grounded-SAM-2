import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
# from PIL import Image
import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, Generator, ItemsView, List, Tuple

def load_first_frame( file_path ):

    data = np.load(file_path, allow_pickle = True)
    data = data.item()
    rgb_np = data['rgb']
    # rgb[...,::-1].copy()
    print("rgb: ", rgb_np.shape)    
    return rgb_np

def load_txt( file_path ):
    # file_path = 'example.txt'
    file_content = None
    with open(file_path, 'r') as file:
        file_content = file.read()

    print(file_content)
    return file_content

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    # a = np.array( mask_util.decode(rle), dtype = np.uint8)
    # print("a: ", a)


    return rle

def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    mask = np.array( mask_util.decode(rle), dtype = np.uint8)
    mask = mask*255
    image = Image.fromarray(mask)

    return image

def main():
    """
    Hyper parameters
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--text-prompt", default="red block. blue block.")
    # parser.add_argument("--img-path", default="notebooks/images/truck.jpg")



    parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    # parser.add_argument("--output-dir", default="outputs/real2sim")

    parser.add_argument("--no-dump-json", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")

    parser.add_argument("-d", "--data_index", default=1,  help="Input data index.")    
    parser.add_argument("-t", "--task", default="hang_mug",  help="Input task name.")
    parser.add_argument("-p", "--project", default="aloha",  help="project name.") 

    # parser.add_argument("--img-path", default="/ws/data/rgb/{}.jpg".format(resolution))
    parser.add_argument("--data-dir-path", default="/ws/data/real2sim/")
    # parser.add_argument("--output-dir", default="/ws/data/real2sim/" )

    args = parser.parse_args()

    GROUNDING_MODEL = args.grounding_model
    TEXT_PROMPT = args.text_prompt
    # IMG_PATH = args.img_path

    SAM2_CHECKPOINT = args.sam2_checkpoint
    SAM2_MODEL_CONFIG = args.sam2_model_config
    DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"

    DUMP_JSON_RESULTS = not args.no_dump_json

    # create output directory
    output_dir = args.data_dir_path + args.task + "/label/" + str( args.data_index )
    OUTPUT_DIR = Path( output_dir )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # environment settings
    # use bfloat16
    torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # build SAM2 image predictor
    sam2_checkpoint = SAM2_CHECKPOINT
    model_cfg = SAM2_MODEL_CONFIG
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino from huggingface
    model_id = GROUNDING_MODEL
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)


    # setup the input image and text prompt for SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot

    # text = TEXT_PROMPT
    # img_path = IMG_PATH
    txt_file_dir = args.data_dir_path + args.task + "/text_prompt.txt"
    text = load_txt(txt_file_dir)

    file_dir = args.data_dir_path + args.task + "/first_frame/" + str( args.data_index ) + ".npy"
    print("processing: ", file_dir)
    rgb_np = load_first_frame(file_dir)

    image = Image.fromarray(rgb_np)
    sam2_predictor.set_image(rgb_np)

    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    """
    Results is a list of dict with the following structure:
    [
        {
            'scores': tensor([0.7969, 0.6469, 0.6002, 0.4220], device='cuda:0'), 
            'labels': ['car', 'tire', 'tire', 'tire'], 
            'boxes': tensor([[  89.3244,  278.6940, 1710.3505,  851.5143],
                            [1392.4701,  554.4064, 1628.6133,  777.5872],
                            [ 436.1182,  621.8940,  676.5255,  851.6897],
                            [1236.0990,  688.3547, 1400.2427,  753.1256]], device='cuda:0')
        }
    ]
    """

    # get the box prompt for SAM 2
    input_boxes = results[0]["boxes"].cpu().numpy()

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )


    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)


    confidences = results[0]["scores"].cpu().numpy().tolist()
    class_names = results[0]["labels"]
    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    """
    Visualize image with supervision useful API
    """
    img = deepcopy( rgb_np[...,::-1].copy() )
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )

    """
    Note that if you want to use default color map,
    you can set color=ColorPalette.DEFAULT
    """
    box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

    mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)

    for idx, mask in enumerate(masks, 0):
        mask = mask.astype(np.uint8)*255
        image = Image.fromarray(mask)
        # Save the image as a grayscale PNG
        # image.save('{}.png'.format(idx))

    """
    Dump the results in standard format and save as json files
    """

    # SAVE_CROPPED_OBJECT = True
    # if SAVE_CROPPED_OBJECT:
    #     for idx, box in enumerate(input_boxes, 0):
    #         print("box: ", box[3] -  box[1], " ", box[2] -  box[0])
    #         cropped_img = rgb_np[int(box[1]) : int(box[3]), int(box[0]) : int(box[2]), :] 
    #         image = Image.fromarray(cropped_img)

    #         # rgb_np[int(box[1]) : int(box[3]), int(box[0]) : int(box[2]), :] = np.array([255,0,0])
    #         # image = Image.fromarray(rgb_np)

    #         # Save the image as a grayscale PNG


    # save results
    # if DUMP_JSON_RESULTS:

    # convert mask into rle format
    mask_rles = [single_mask_to_rle(mask) for mask in masks]
    uncompressed_masks = [ rle_to_mask(rle) for rle in mask_rles ]

    labels = []
    binary_masks = []

    input_boxes = (input_boxes)
    int_boxes = []
    for idx, mask in enumerate(uncompressed_masks, 0):
        binary_mask = np.asarray(mask) // 255
        
        invalid_idx = np.where(binary_mask == 0)
        # print("invalid_idx: ", len(invalid_idx))
        image = rgb_np
        image[invalid_idx] = np.array([255,255,255])
        # image = Image.fromarray(image)
        box = input_boxes[idx]
        print("box: ", box)
        print("int box: ", [ int(box[0]),int(box[1]), int(box[2]), int(box[3]) ] )
        cropped_img = image[int(box[1]) : int(box[3]), int(box[0]) : int(box[2]), :]
        int_boxes.append( [ int(box[0]),int(box[1]), int(box[2]), int(box[3]) ] )
        image = Image.fromarray(cropped_img)
        # image = image.resize( (256,256) )
        # image.save( os.path.join(OUTPUT_DIR, '{}.png'.format(idx+1)) )
        # Save the image as a grayscale PNG
        mask.save( os.path.join(OUTPUT_DIR, "mask_{}.png".format(idx) ) )

        binary_masks.append(binary_mask)

    numpy_dict = {}
    numpy_dict['binary_mask'] = binary_masks
    numpy_dict['class_names'] = class_names    
    numpy_dict['scores'] = scores  
    numpy_dict['boxes'] = int_boxes
    numpy_dict['box_format'] = "image[ box[1]:box[3], box[0]:box[2] ]"
    np.save(os.path.join( OUTPUT_DIR, "data" ), numpy_dict)



    input_boxes = input_boxes.tolist()
    scores = scores.tolist()
    # save the results in standard format
    results = {
        "image_path": file_dir,
        "annotations" : [
            {
                "class_name": class_name,
                "bbox": box,
                "segmentation": mask_rle,
                "score": score,
            }
            for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
        ],
        "box_format": "xyxy",
        "img_width": image.width,
        "img_height": image.height,
    }

    # save a numpy
    # for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
    
    with open(os.path.join(OUTPUT_DIR, "grounded_sam2_hf_model_demo_results.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()