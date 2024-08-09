import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def convert_annotation(xml_file, txt_file, image_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    image_filename = root.find('filename').text
    image_path = os.path.join(image_dir, image_filename)
    
    # Get image size
    with Image.open(image_path) as img:
        width, height = img.size
    
    with open(txt_file, 'w') as f:
        for obj in root.findall('object'):
            ii = obj.find('name')
            if ii is None:
                print(f"Skipping {xml_file} as it has no name tag")
                continue

            cls = ii.text
            if cls != 'head':  # Ensure to match the label in your dataset
                continue

            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            ymin = float(xmlbox.find('ymin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymax = float(xmlbox.find('ymax').text)
            x_center = (xmin + xmax) / 2.0 / width
            y_center = (ymin + ymax) / 2.0 / height
            w = (xmax - xmin) / float(width)
            h = (ymax - ymin) / float(height)
            f.write(f"0 {x_center} {y_center} {w} {h}\n")

def convert_dataset(xml_dir, txt_dir, image_dir):
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    
    xml_files = [os.path.join(xml_dir, f) for f in os.listdir(xml_dir) if f.endswith('.xml')]
    tasks = []
    
    with ThreadPoolExecutor() as executor:
        for xml_file in tqdm(xml_files):
            base_name = os.path.splitext(os.path.basename(xml_file))[0]
            txt_file = os.path.join(txt_dir, base_name + '.txt')
            tasks.append(executor.submit(convert_annotation, xml_file, txt_file, image_dir))
        
        for task in tqdm(tasks):
            task.result()  # Wait for all tasks to complete

# Paths to your dataset
xml_dir = './Annotations'  # Directory with annotation XML files
txt_dir = './labels/all'  # Directory where YOLO format labels will be saved
image_dir = './JPEGImages'  # Directory with images

convert_dataset(xml_dir, txt_dir, image_dir)
