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
* 2: Label plaatjes in de braincreator tool of zorg voor annotaties in VOC format(xml). Met Braincreator: exporteer de coco labels naar de DGX. Noem het labels.json en zet het in je main_dir. Anders: zet annotaties in main_dir/raw/Annotations. Let er op dat de image paden in de annotaties naar het goede bestand verwijzen. Vanuit Braincreator verwijzen alle xml's naar de map raw/Images.
* 3: Maak een classes.names bestand. Hierin stop je de classes die je geannoteerd hebt in dezelfde volgorde als in de labeltool. Zet die ook in je main_dir.
* 4: Zet al je images in main_dir/raw/. Het model maakt een directory raw/Images en vult die met alleen de images die ook een annotatie hebben. Images die na het preprocessen nog los in raw staan worden niet gebruikt voor het trainen, want daar is geen annotatie voor. Indien de plaatjes anders zijn dan .png, geef bij het preprocessen de flag --img_type .XXX aan
* 5: Draai RWS_Yolo.py met `--preprocessing True. Het labels.json bestand wordt geconverteerd naar xml's, de juiste images worden erbij gezocht en worden in de mappen raw/Annotations of raw/Images gezet. Deze mappen worden random in een train/validation set gemaakt en geconverteerd naar TFRecord format voor het trainen. 
* 6: Kies je trainings parameters of default met `RWS_Yolo.py --train True`
* 7: Model getraind? Evalueren of predicten met de bijbehorende flags. 

Voor nu moet je het script aanroepen in de map /computer-vision/ADR/scripts_full_pipeline met `python RWS_Yolo.py` en de bijbehorende vlaggen --FLAGNAME. Zie `python RWS_YOlo.py --helpshort` voor meer info
