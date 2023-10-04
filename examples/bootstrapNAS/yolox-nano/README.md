# YoloX-NAS

----
### Prepare Dataset 
```
cd /data/dataset/
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar xf VOCtrainval_06-Nov-2007.tar
mkdir voc2007_coco
cd voc2007_coco

wget https://raw.githubusercontent.com/yukkyo/voc2coco/master/voc2coco.py
mkdir annotations
ln -s ../VOCdevkit/VOC2007/Annotations .
ln -s ../VOCdevkit/VOC2007/ImageSets .
```

Add class_list.txt to this directory.

```
python3 voc2coco.py --ann_dir Annotations/ --ann_ids ImageSets/Main/train.txt --labels class_list.txt --output annotations/instances_train.json --ext xml --extract_num_from_imgid
python3 voc2coco.py --ann_dir Annotations/ --ann_ids ImageSets/Main/val.txt   --labels class_list.txt --output annotations/instances_val.json   --ext xml --extract_num_from_imgid

ln -s ../VOCdevkit/VOC2007/JPEGImages/ train2017
ln -s ../VOCdevkit/VOC2007/JPEGImages/ val2017

**change to working directory.**

git clone https://github.com/Megvii-BaseDetection/YOLOX.git && cd YOLOX
git checkout -b bootstrapnas bb9185c095dfd7a8015a1b82f3e9a065090860b8
git apply < /path/to/yolox-bootstrapnas.patch

cd datasets && ln -s ../../VOCdevkit/VOC2007 && cd -
cd datasets && ln -s /data/dataset/voc2007_coco/ VOC2007 && cd -

poetry config --local virtualenvs.in-project true
poetry install
poetry shell


#train without BootstrapNAS to get pretrained weight
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth
C=yolox_nano.pth
PYTHONPATH=. nohup 2>&1 python tools/train.py -f exps/default/yolox_nano_voc-e50.py -d 1 -b 6 -o -c $C --cache

#train with BootstrapNAS using pretrained initial weight
C=YOLOX_outputs/yolox_nano_voc-e50/best_ckpt.pth
PYTHONPATH=. nohup 2>&1 python tools/train.py -f exps/default/first_try.py -d 1 -b 6 -o -c $C --cache --nncf_config_path nncf_config_yolox_bootstrapNAS.json
```
