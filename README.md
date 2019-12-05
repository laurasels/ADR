# ADR
## Herkennen stickers gevaarlijke stoffen op vrachtwagens

De beeldherkenning zal gebaseerd zijn op deze code: https://github.com/zzh8829/yolov3-tf2/blob/master/train.py (Tensorflow 2.0, Yolo3)

## To do:
* Opknippen videomateriaal in afzonderlijke frames: `vlc "A2 HBR 59 92.wmv" --video-filter=scene --vout=dummy --scene-ratio=10 --scene-path="/home/datalab/Documenten/ADR/frames" vlc://quit`

## Stap 1: Model vrachtwagens
* Labelen vrachtwagens (labeltool)
* Model trainen om vrachtwagens te herkennen
* Model testen 

## Stap 2: Model borden
* Labelen ADR-borden op vrachtwagens (labeltool)
* Model trainen om borden op vrachtwagens te herkennen
* Model testen

## Stap 3: Model tekst op borden
* Techniek (OCR) gebruiken om codes op herkende borden te lezen 
