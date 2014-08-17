# -*- coding: cp936-*-
"""KIT Parser (Python version)
Extract the KIT data sets information from the xml files.
"""
__author__ = 'shuaiyi'

import os
from BeautifulSoup import BeautifulSoup


class KIT:
    def __init__(self, dataset_path):
        self.workspace = dataset_path
        self.regions = {}
        self.num = 0

    def gen_regions(self):
        """通过工程目录获取数据整体结构
        :param dataset_path: 数据根目录
        """
        for path, dirs, files in os.walk(self.workspace):
            for f in files:
                if f.find('.xml') != -1:
                    self.regions[f[0:-4]] = {'path': os.path.join(path, f),
                                             'frames': {}}
        self.num=len(self.regions)

    def get_car_list(self, region_name):
        """获取单个区域的信息
        :param region_name: 区域名称
        """
        xml = BeautifulSoup(open(self.regions[region_name]['path']))
        all_frames = xml.dataset.findAll('frame')
        object_list = xml.dataset.findAll('objectlist')
        for i in range(len(all_frames)):
            frame = {}
            frame['img']=all_frames[i]['file']
            frame['cars']=list()
            #img = cv2.imread(frames[i]['file'])
            objects = object_list[i]
            for car in objects.findAll('object'):
                objectid = int(car['id'])
                xc = float(car.box['xc'])
                yc = float(car.box['yc'])
                w = float(car.box['w'])
                h = float(car.box['h'])
                o = float(car.representation['o'])
                frame['cars'].append({'id': objectid, 'xc': xc, 'yc': yc, 'w': w, 'h': h, 'o': o})
            self.regions[region_name]['frames'][i]=frame

    def extract_all(self):
        """提取所有
        """
        print 'Init kit...'
        self.gen_regions()
        for region in self.regions.keys():
            # print region
            print 'parse %s xml file' % region
            self.get_car_list(region)

if __name__ == '__main__':
    # todo verify the class kit
    WorkSpace='E:\\2013\\Samples-for-cuda-convnet\\Training'
    kit=KIT(WorkSpace)
    kit.extract_all()
    print kit.regions