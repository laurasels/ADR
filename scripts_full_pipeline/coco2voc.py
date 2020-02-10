'''Transforms coco-json output from brainmatter to VOC xml files for trainging
'''
import pandas as pd
from pathlib import Path
import os
import json
from absl import app, flags, logging
from absl.flags import FLAGS
from shutil import rmtree
import sys


def writebbox2xml(bbox,outputDir,file,annotation_name,width,height):
    """
    Actual writing of the xml file
    
    Parameters
    ----------
    bbox               list of bbox coordinates [xmin,ymin,xmax,ymax]
    outputdir          location where the final xml will be saved
    file               name of the image file. Will also be the name of the xml (minus .png)
    annotation_name    name of the labeled object
    width/height       width/height of the image

    Returns
    -------
    Returns nothing, but saves the xml
    """
    outputDir = Path(outputDir)
    # --  Write output of boundingbox to xml file
    outputPath = outputDir / file.split('/')[-1].replace('.png', '.xml')
    
    # If file exists, then append the other object from that picture
    if os.path.exists(outputPath):
        # Read in the file
        with open(outputPath, 'r') as the_file :
          filedata = the_file.read()
        
        # Replace the last line that closes the annotation file
        filedata = filedata.replace('\n</annotation>', '')
        
        # Write the file out again and replace the first part with the new text
        with open(outputPath, 'w') as the_file:
            the_file.write(filedata)
            the_file.write('\n\t<object>')
            the_file.write('\n\t\t<name>%s</name>' %(annotation_name))
            the_file.write('\n\t\t<pose>Unspecified</pose>')
            the_file.write('\n\t\t<truncated>0</truncated>')
            the_file.write('\n\t\t<difficult>0</difficult>')
            the_file.write('\n\t\t<bndbox>')
            the_file.write('\n\t\t\t<xmin>%s</xmin>'%(bbox[0]))
            the_file.write('\n\t\t\t<ymin>%s</ymin>'%(bbox[1]))
            the_file.write('\n\t\t\t<xmax>%s</xmax>'%(bbox[2]))
            the_file.write('\n\t\t\t<ymax>%s</ymax>'%(bbox[3]))
            the_file.write('\n\t\t</bndbox>')
            the_file.write('\n\t</object>')
            the_file.write('\n</annotation>')
    else:    
        with open(str(outputPath), 'w') as the_file:
            the_file.write('<annotation verified="yes">')
            the_file.write('\n\t<folder>images</folder>')
            the_file.write('\n\t<filename>%s</filename>' %(file))
            the_file.write('\n\t<path>%s </path>'%(file)) # MAG NIET LEEG ZIJN
            the_file.write('\n\t<source>')
            the_file.write('\n\t\t<database>Unknown</database>')
            the_file.write('\n\t</source>')
            the_file.write('\n\t<size>')
            the_file.write('\n\t\t<width>%s</width>' %(width))
            the_file.write('\n\t\t<height>%s</height>' %(height))
            the_file.write('\n\t\t<depth>3</depth>')
            the_file.write('\n\t</size>')
            the_file.write('\n\t<segmented>0</segmented>')
            
            the_file.write('\n\t<object>')
            the_file.write('\n\t\t<name>%s</name>' %(annotation_name))
            the_file.write('\n\t\t<pose>Unspecified</pose>')
            the_file.write('\n\t\t<truncated>0</truncated>')
            the_file.write('\n\t\t<difficult>0</difficult>')
            the_file.write('\n\t\t<bndbox>')
            the_file.write('\n\t\t\t<xmin>%s</xmin>'%(bbox[0]))
            the_file.write('\n\t\t\t<ymin>%s</ymin>'%(bbox[1]))
            the_file.write('\n\t\t\t<xmax>%s</xmax>'%(bbox[2]))
            the_file.write('\n\t\t\t<ymax>%s</ymax>'%(bbox[3]))
            the_file.write('\n\t\t</bndbox>')
            the_file.write('\n\t</object>')
            the_file.write('\n</annotation>')

def coco2voc():
    """
    Transforms coco-json output from brainmatter to VOC xml files for trainging

    Parameters
    ----------
    Flags.main_dir         the main directory of the project
    
    Returns
    -------
    xml file for each image in the annotated image
    """
    # Read coco.json from Brainmatter    
    with open(os.path.join(FLAGS.main_dir, FLAGS.json_file)) as handle:
        coco = json.loads(handle.read())

    img_dir = os.path.join(FLAGS.main_dir, 'raw/Images/')
    xml_dir = os.path.join(FLAGS.main_dir, 'raw/Annotations')
    
    # Check if xml_dir, if so, empty the directory
    if os.path.exists(xml_dir):
        rmtree(xml_dir)
        os.makedirs(xml_dir)


    # Create category df    
    category_ids = []
    category_names = []
    for i in range(len(coco['categories'])):
        print(i)
        category_ids.append(coco['categories'][i]['id'])
        category_names.append(coco['categories'][i]['name'])
        
    category_df = pd.DataFrame({'names':category_names},index=category_ids)
    
    # Create image df
    image_ids = []
    image_names = []
    image_widths = []
    image_heights = []
    for i in range(len(coco['images'])):
        print(i)
        image_ids.append(coco['images'][i]['id'])
        filename = coco['images'][i]['file_name']
        image_names.append(img_dir + filename.split('/')[-1])
        image_heights.append(coco['images'][i]['height'])
        image_widths.append(coco['images'][i]['width'])
        
    image_df = pd.DataFrame({'filename':image_names,
                             'height':image_heights,
                             'width':image_widths},index=image_ids)
    
    # Loop over annotations, read info from the category and image df and
    # write the output as a bbox in in an xml file
    for i in range(len(coco['annotations'])):
        image_id = coco['annotations'][i]['image_id']
        image_name = image_df.loc[image_id,'filename']
        image_height = image_df.loc[image_id,'height']
        image_width = image_df.loc[image_id,'width']
        annotation_id = coco['annotations'][i]['category_id']
        annotation_name = category_df.loc[annotation_id,'names']
        
        bbox = coco['annotations'][i]['bbox']
        
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[0]+bbox[2]
        ymax = bbox[1]+bbox[3]
        
        bbox = [xmin,ymin,xmax,ymax]
        print(image_name)
        writebbox2xml(bbox,xml_dir,image_name,annotation_name,image_width,image_height)
  
    
