## Требования
* python3.10
* numpy
* tensorflow2.15
* opencv2
* cuda 11.1
* cudnn 8.0

Так как после 2.15 tensorflow перешёл в tensorflow-intel, который не поддерживает gpu на windows, то проект временно не будет портироваться на более новые версии библиотек

## Тестовые примеры:

`python3 run.py` - для запуска тестового изображения

`python3 run.py -v 0` - для запуска на вебкамере

`python3 run.py -v video.mp4` - для запуска на видео
