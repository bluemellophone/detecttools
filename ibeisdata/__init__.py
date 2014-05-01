#!/usr/bin/env python

from datetime import date
import cv2
import numpy as np
import os
import random
from directory import Directory
import xml.etree.ElementTree as xml


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def _kwargs(kwargs, key, value):
    if key not in kwargs.keys():
        kwargs[key] = value


def get(et, category, text=True, singularize=True):
    temp = [(_object.text if text else _object) for _object in et.findall(category)]
    if len(temp) == 1 and singularize:
        temp = temp[0]
    return temp


def histogram(_list):
    retDict = {}
    for value in _list:
        if value in retDict:
            retDict[value] += 1
        else:
            retDict[value] = 1
    return retDict


def openImage(filename, color=False, alpha=False):
    if not os.path.exists(filename):
        return None

    if not color:
        mode = 0 # Greyscale by default
    elif not alpha:
        mode = 1 # Color without alpha channel
    else:
        mode = -1 # Color with alpha channel

    return cv2.imread(filename, mode)


def randInt(lower, upper):
    return random.randint(lower, upper)


def randColor():
    return [randInt(50, 205), randInt(50, 205), randInt(50, 205)]


class IBEIS_Data(object):

    def __init__(ibsd, dataset_path, **kwargs): 
        _kwargs(kwargs, 'object_min_width',    32)
        _kwargs(kwargs, 'object_min_height',   32)
        _kwargs(kwargs, 'mine_negatives',      True)
        _kwargs(kwargs, 'mine_width_min',      50)
        _kwargs(kwargs, 'mine_width_max',      400)
        _kwargs(kwargs, 'mine_height_min',     50)
        _kwargs(kwargs, 'mine_height_max',     400)
        _kwargs(kwargs, 'mine_max_attempts',   100)
        _kwargs(kwargs, 'mine_max_keep',       10)
        _kwargs(kwargs, 'mine_overlap_margin', 0.25)
        _kwargs(kwargs, 'mine_exclude_categories', [])

        ibsd.dataset_path = dataset_path
        ibsd.absolute_dataset_path = os.path.realpath(dataset_path)

        direct = Directory(os.path.join(dataset_path, "Annotations") , include_file_extensions=["xml"])
        ibsd.images = []
        files = direct.files()
        print "Loading Database"
        for i, filename in enumerate(files):
            if len(files) > 10:
                if i % (len(files) / 10) == 0:
                    print "%0.2f" %(float(i) / len(files))
            ibsd.images.append(IBEIS_Image(filename, ibsd.absolute_dataset_path, **kwargs))
        print "    ...Loaded"

        ibsd.categories_images = []
        ibsd.categories_rois = []

        for image in ibsd.images:
            temp = image.categories(unique=False)
            ibsd.categories_rois += temp
            ibsd.categories_images += set(temp)

            if len(image.objects) == 0:
                ibsd.categories_images += ["IGNORED"]

        ibsd.distribution_images = histogram(ibsd.categories_images)
        ibsd.distribution_rois = histogram(ibsd.categories_rois)
        ibsd.rois = sum(ibsd.distribution_rois.values())
        
        ibsd.categories = sorted(set(ibsd.categories_images))
    
    def __str__(ibsd):
        return "<IBEIS Data Object | %s | %d images | %d categories | %d rois>" \
            %(ibsd.absolute_dataset_path, len(ibsd.images), len(ibsd.categories), ibsd.rois)

    def __repr__(ibsd):
        return "<IBEIS Data Object | %s>" % (ibsd.absolute_dataset_path)

    def __len__(ibsd):
        return len(ibsd.images)

    def __getitem__(ibsd, key):
        if isinstance(key, str):
            for image in ibsd.images:
                if key in image.filename:
                    return image
            return None
        else:
            return ibsd.images[key]

    def print_distribution(ibsd):
        def _print_line(category, spacing, images, rois):
            images = str(images)
            rois = str(rois)
            print "%s%s\t%s" %(category + " " * (spacing - len(category)), images, rois)
        
        _max = max([ len(category) for category in ibsd.distribution_rois.keys() + ['TOTAL', 'CATEGORY'] ]) + 3

        _print_line("CATEGORY", _max, "IMGs", "ROIs")
        if "IGNORED" in ibsd.distribution_images:
            _print_line("IGNORED", _max, ibsd.distribution_images["IGNORED"], "")

        for category in sorted(ibsd.distribution_rois):
            _print_line(category, _max, ibsd.distribution_images[category], ibsd.distribution_rois[category])
        
        _print_line("TOTAL", _max, len(ibsd.images), ibsd.rois)

    def dataset(ibsd, positive_category, negative_exclude_categories=[], max_rois_pos=None, max_rois_neg=None):
        def _parse_dataset_file(category, _type):
            filepath = os.path.join(ibsd.dataset_path, "ImageSets", "Main", category + "_" + _type + ".txt")
            _dict = {}
            try:
                _file = open(filepath)
                for line in _file:
                    line = line.strip().split(" ")
                    line = [line[0], line[-1]]
                    _dict[line[0]] = True
            except IOError as e:
                print "<", e, ">", filepath

            return _dict

        positives = []
        negatives = []
        validation = []
        test = []

        train_values = _parse_dataset_file(positive_category, "train")
        val_values = _parse_dataset_file(positive_category, "val")
        test_values = _parse_dataset_file(positive_category, "test")
        
        pos_rois = 0
        neg_rois = 0
        for image in ibsd.images:
            filename = image.filename
            _train = train_values.get(image.filename[:-4], False)
            _val = val_values.get(image.filename[:-4], False)
            _test = test_values.get(image.filename[:-4], False)

            temp = image.categories(unique=False)
            flag = False

            if _train:
                for val in temp:
                    if val == positive_category:
                        flag = True
                        pos_rois += 1
                    elif val not in negative_exclude_categories:
                        neg_rois += 1

                if flag:
                    positives.append(image)
                elif val not in negative_exclude_categories:
                    negatives.append(image)

            if _val:
                validation.append(image)

            if _test:
                test.append(image)

        # Setup auto normalize variables for equal positives and negatives
        if max_rois_pos == -1:
            max_rois_pos = neg_rois

        if max_rois_neg == -1:
            max_rois_neg = pos_rois

        # Remove positives to target, not gauranteed to give target, but 'close'.
        if max_rois_pos is not None and len(positives) > 0:
            pos_density = float(pos_rois) / len(positives)
            target_num = int(max_rois_pos / pos_density)
            print "Normalizing Positives, Target:", target_num
            
            # Remove images to match target
            while len(positives) > target_num:
                positives.pop( randInt(0, len(positives) - 1) )

            # Recalculate rois left
            pos_rois = 0
            for image in positives:
                temp = image.categories(unique=False)
                for val in temp:
                    if val in positive_category:
                        pos_rois += 1

        # Remove positives to target, not gauranteed to give target, but 'close'.
        if max_rois_neg is not None and len(negatives) > 0:
            neg_density = float(neg_rois) / len(negatives)
            target_num = int(max_rois_neg / neg_density)
            print "Normalizing Negatives, Target:", target_num
            
            # Remove images to match target
            while len(negatives) > target_num:
                negatives.pop( randInt(0, len(negatives) - 1) )

            # Recalculate rois left
            neg_rois = 0
            for image in negatives:
                temp = image.categories(unique=False)
                for val in temp:
                    if val not in positive_category:
                        neg_rois += 1

        return (positives, pos_rois), (negatives, neg_rois), validation, test


class IBEIS_Image(object):

    def __init__(ibsi, filename_xml, absolute_dataset_path, **kwargs):
        with open(filename_xml, 'r') as _xml:
            _xml = xml.XML(_xml.read().replace('\n', ''))
            
            ibsi.folder = get(_xml, 'folder')
            ibsi.absolute_dataset_path = absolute_dataset_path
            ibsi.filename = get(_xml, 'filename')

            source = get(_xml, 'source', text=False)
            ibsi.source_database = get(source, 'database')
            ibsi.source_annotation = get(source, 'annotation')
            ibsi.source_image = get(source, 'image')

            size = get(_xml, 'size', text=False)
            ibsi.width = int(get(size, 'width'))
            ibsi.height = int(get(size, 'height'))
            ibsi.depth = int(get(size, 'depth'))

            ibsi.segmented = get(size, 'segmented') == "1"

            ibsi.objects = [ IBEIS_Object(obj, ibsi.width, ibsi.height) for obj in get(_xml, 'object', text=False, singularize=False) ]
            
            for _object in ibsi.objects:
                if _object.width <= kwargs['object_min_width'] or \
                   _object.height <= kwargs['object_min_height']:
                    # Remove objects that are too small.
                    ibsi.objects.remove(_object)

            flag = True
            for cat in ibsi.categories():
                if cat in kwargs['mine_exclude_categories']:
                    flag = False
                
            if kwargs['mine_negatives'] and flag:

                def _overlaps(objects, obj, margin):
                    for _obj in objects:
                        leftA   = obj['xmin']
                        rightA  = obj['xmax']
                        bottomA = obj['ymin']
                        topA    = obj['ymax']
                        widthA = rightA - leftA
                        heightA = topA - bottomA

                        leftB   = _obj.xmin + 0.25 * min(_obj.width, widthA)
                        rightB  = _obj.xmax - 0.25 * min(_obj.width, widthA)
                        bottomB = _obj.ymin + 0.25 * min(_obj.height, heightA)
                        topB    = _obj.ymax - 0.25 * min(_obj.height, heightA)

                        if (leftA < rightB) and (rightA > leftB) and \
                           (topA > bottomB) and (bottomA < topB):
                            return True

                    return False

                negatives = 0
                for i in range(kwargs['mine_max_attempts']):
                    if negatives >= kwargs['mine_max_keep']:
                        break

                    width = randInt(kwargs['mine_width_min'], min(ibsi.width - 1, kwargs['mine_width_max']))
                    height = randInt(kwargs['mine_height_min'], min(ibsi.height - 1, kwargs['mine_height_max']))
                    x = randInt(0, ibsi.width - width - 1)
                    y = randInt(0, ibsi.height - height - 1)
                    
                    obj = {
                        'xmax': x + width,
                        'xmin': x,
                        'ymax': y + height,
                        'ymin': y,
                    }

                    if _overlaps(ibsi.objects, obj, kwargs["mine_overlap_margin"]):
                        continue

                    ibsi.objects.append(IBEIS_Object(obj, ibsi.width, ibsi.height, implicit=False))
                    negatives += 1


    def __str__(ibsi):
        return "<IBEIS Image Object | %s | %d objects>" \
            %(ibsi.filename, len(ibsi.objects))

    def __repr__(ibsi):
        return "<IBEIS Image Object | %s>" % (ibsi.filename)

    def __len__(ibsi):
        return len(ibsi.objects)

    def image_path(ibsi):
        return os.path.join(ibsi.absolute_dataset_path, "JPEGImages", ibsi.filename)

    def categories(ibsi, unique=True):
        temp = [ _object.name for _object in ibsi.objects ]
        if unique:
            temp = set(temp)
        return sorted(temp)

    def bounding_boxes(ibsi, parts=False):
        return [ _object.bounding_box(parts) for _object in ibsi.objects ]

    def show(ibsi, objects=True, parts=True, display=True):

        def _draw_box(img, annotation, xmin, ymin, xmax, ymax, color):
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5

            width, height = cv2.getTextSize(annotation, font, scale, -1)[0]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.rectangle(img, (xmin, ymin - height), (xmin + width, ymin), color, -1)
            cv2.putText(img, annotation, (xmin + 5, ymin), font, 0.4, (255, 255, 255))

        original = openImage(ibsi.image_path(), color=True)
        
        for _object in ibsi.objects:
            color = randColor()
            _draw_box(original, _object.name.upper(), _object.xmin, _object.ymin, _object.xmax, _object.ymax, color)

            if parts:
                for part in _object.parts:
                    _draw_box(original, part.name.upper(), part.xmin, part.ymin, part.xmax, part.ymax, color)

        if display:
            cv2.imshow(ibsi.filename + " with Bounding Boxes", original)
            cont = raw_input()
            cv2.destroyAllWindows()
            return cont == ""
        else:
            return original
    
    

class IBEIS_Object(object):

    def __init__(ibso, _xml, width, height, implicit=True, **kwargs):
        if implicit:
            ibso.name = get(_xml, 'name')
            ibso.pose = get(_xml, 'pose')
            ibso.truncated = get(_xml, 'truncated') == "1"
            ibso.difficult = get(_xml, 'difficult') == "1"
                
            bndbox = get(_xml, 'bndbox', text=False)
            ibso.xmax = min(width,  int(float(get(bndbox, 'xmax'))))
            ibso.xmin = max(0,      int(float(get(bndbox, 'xmin'))))
            ibso.ymax = min(height, int(float(get(bndbox, 'ymax'))))
            ibso.ymin = max(0,      int(float(get(bndbox, 'ymin'))))

            ibso.parts = [ IBEIS_Part(part) for part in get(_xml, 'part', text=False, singularize=False)]
        else:
            ibso.name = 'MINED'
            ibso.pose = 'Unspecified'
            ibso.truncated = False
            ibso.difficult = False
                
            ibso.xmax = min(width,  _xml['xmax'])
            ibso.xmin = max(0,      _xml['xmin'])
            ibso.ymax = min(height, _xml['ymax'])
            ibso.ymin = max(0,      _xml['ymin'])

            ibso.parts = []

        ibso.width = ibso.xmax - ibso.xmin
        ibso.height = ibso.ymax - ibso.ymin
        ibso.area = ibso.width * ibso.height


    def __len__(ibso):
        return len(ibso.parts)

    def bounding_box(ibso, parts=False):
        _parts = [ part.bounding_box() for part in ibso.parts ]
        return [ibso.name, ibso.xmax, ibso.xmin, ibso.ymax, ibso.ymin, _parts]

class IBEIS_Part(object):

    def __init__(ibsp, _xml, **kwargs):
        ibsp.name = get(_xml, 'name')
            
        bndbox = get(_xml, 'bndbox', text=False)
        ibsp.xmax = int(float(get(bndbox, 'xmax')))
        ibsp.xmin = int(float(get(bndbox, 'xmin')))
        ibsp.ymax = int(float(get(bndbox, 'ymax')))
        ibsp.ymin = int(float(get(bndbox, 'ymin')))
        ibsp.width = ibsp.xmax - ibsp.xmin
        ibsp.height = ibsp.ymax - ibsp.ymin
        ibsp.area = ibsp.width * ibsp.height

    def bounding_box(ibsp):
        return [ibsp.name, ibsp.xmax, ibsp.xmin, ibsp.ymax, ibsp.ymin]


if __name__ == "__main__":

    information = {
        'mine_negatives':   True,
        'mine_max_keep':    1,
        'mine_exclude_categories': ['zebra_grevys', 'zebra_plains'],
    }

    dataset = IBEIS_Data('test/', **information)
    # dataset = IBEIS_Data('/Datasets/VOC2012/', **information)
    print dataset
    print 

    # Access specific information about the dataset
    print "Categories:", dataset.categories
    print "Number of images:", len(dataset)

    print 
    dataset.print_distribution()
    print 

    # Access specific image from dataset using filename or index
    print dataset['2014_000002']
    print dataset['_000002'] #partial also works (takes first match)
    cont = True
    while cont:
        # Show the detection regions by drawing them on the source image
        print "Enter something to continue, empty to get new image"
        cont = dataset[randInt(0, len(dataset) - 1)].show()

    # Get all images using a specific positive set
    # (pos, pos_rois), (neg, neg_rois), val, test = dataset.dataset('chair', max_rois_neg=-1)
    (pos, pos_rois), (neg, neg_rois), val, test = dataset.dataset('zebra_grevys')
    # (pos, pos_rois), (neg, neg_rois), val, test = dataset.dataset('zebra_grevys', max_rois_neg=-1)
    print "%s\t%s\t%s\t%s\t%s" %("       ", "Pos", "Neg", "Val", "Test")
    print "%s\t%s\t%s\t%s\t%s" %("Images:", len(pos), len(neg), len(val), len(test))
    print "%s\t%s\t%s\t%s\t%s" %("ROIs:  ", pos_rois, neg_rois, "", "")

    # print "\nPositives:"
    # for _pos in pos:
    #     print _pos.image_path()
    #     print _pos.bounding_boxes(parts=True)

    # print "\nNegatives:"
    # for _neg in neg:
    #     print _neg.image_path()
    #     print _neg.bounding_boxes(parts=True)

    # print "\nValidation:"
    # for _val in val:
    #     print _val.image_path()
    #     print _val.bounding_boxes(parts=True)

    # print "\nTest:"
    # for _test in test:
    #     print _test.image_path()
    #     print _test.bounding_boxes(parts=True)
