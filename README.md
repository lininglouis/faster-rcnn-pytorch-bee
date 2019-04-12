 A faster rcnn pytorch version for bee
 
 original repo: https://github.com/jwyang/faster-rcnn.pytorch
 
 ### problem 1
 when coming across problem "no module named _mask"
 you need to install the latest cocoapi by
 ```shell
$ git clone https://github.com/pdollar/coco.git
$ cd coco/PythonAPI
$ make
$ python setup.py install

then change the old cocoapi to your built latest cocoapi
$ cd lib
$ mv pycocotools/ pycocotools_old
$ ln -s <PATH_TO_FRESH_GIT_CLONE>/cocoapi/PythonAPI/pycocotools pycocotools

 ```
 
 
### step to run
 0) compile the dependencies
```shell
$ cd lib
$ python setup.py build develop
```
 1) fill the colony with images
 2) add pth to the model folder(saved in baidu pan 156)
 3) run train.sh
