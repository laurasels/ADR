# ADR
## Herkennen stickers gevaarlijke stoffen op vrachtwagens

De beeldherkenning zal gebaseerd zijn op deze code: https://github.com/zzh8829/yolov3-tf2/blob/master/train.py (Tensorflow 2.0, Yolo3)

## To do:
* Opknippen videomateriaal in afzonderlijke frames: 
`vlc "filename.mp4" --video-filter=scene --vout=dummy --scene-ratio=10 --scene-path="/home/datalab/Documenten/ADR/frames" vlc://quit` 

### Stap 1: Model vrachtwagens
* Voorgetraind model Yolo3 gebruiken om vrachtwagens te herkennen, dit kan ook gelijk vanaf de video:
`detect_video3.py --video /home/datalab/Documenten/ADR/Beneluxtunnel/filename.mp4 --output ./output.avi > output.txt`
* Frames vrachtwagens wegschrijven naar aparte mappen, met eventueel txt bestandje erbij. 

### Stap 2: Model borden
* Labelen ADR-borden op vrachtwagens in de map (labeltool)
* Model trainen om borden op vrachtwagens te herkennen
* Model testen

### Stap 3: Model tekst op borden
* Techniek (OCR) gebruiken om codes op herkende borden te lezen 

