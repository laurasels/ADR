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


### RWS Yolo handleiding
* 1: Maak een map op de DGX. Dit wordt je 'main_dir'
* 2: Label plaatjes in de braincreator tool of zorg voor annotaties in VOC format(xml). Met Braincreator: exporteer de coco labels naar de DGX. Noem het labels.json en zet het in je main_dir. Anders: zet annotaties in main_dir/raw/Annotations. Let er op dat de image paden in de annotaties naar het goede bestand verwijzen. 
* 3: Maak een classes.names bestand. Hierin stop je de classes die je geannoteerd hebt in dezelfde volgorde als in de labeltool. Zet die ook in je main_dir.
* 4: Zet al je images in main_dir/raw/Images. Die gebruikt het model om te trainen.
* 5: Draai RWS_Yolo.py met `--preprocessing True`
* 6: Kies je trainings parameters of default met `RWS_Yolo.py --train True`
* 7: Model getraind? Evalueren of predicten met de bijbehorende flags. 

