 A faster rcnn pytorch version for bee
 
 original repo: https://github.com/jwyang/faster-rcnn.pytorch
 
 when coming across problem "no module named _mask"
 you need to install the latest cocoapi by
 ```shell
$ git clone https://github.com/pdollar/coco.git
$ cd coco/PythonAPI
$ make
$ python setup.py install
 ```
 
 
 step to run
 
 1) fill the colony with images
 2) add pth to the model folder(saved in baidu pan 156)
 3) run train.sh
