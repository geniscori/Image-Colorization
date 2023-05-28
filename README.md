# XNAP-Colorització d'imatges en blanc i negre utilitzant CNN

##  Introducció

Aquest projecte utilitza Xarxes Neuronals Convolucionals (CNN) per transformar imatges en escala de grisos en representacions en color.

## Objectius
El principal objectiu d'aquest projecte és entendre i saber desenvolupar una solució basada en CNN que prengui imatges en escala de grisos com a entrada i generi versions en color corresponents. Mitjançant la formació de la nostra xarxa amb un gran conjunt de dades d'imatges en color, pretenem permetre que entengui les relacions complexes entre els valors d'intensitat en escala de grisos i els seus colors corresponents.

Els objectius del nostre projecte inclouen:

- Entendre el funcionament de l'arquitectura CNN i com aplicar-lo en la colorització d'imatges.
- Construir una arquitectura de CNN de principi a fi capaç d'aprendre representacions de color significatives a partir d'imatges en escala de grisos.
- Entrenar la xarxa amb un conjunt divers de dades d'imatges en escala de grisos i en color corresponents per capturar diversos patrons i matisos de color.

## Prerequisits
Per tal de poder executar el codi, s’han de tenir les següents dependències i paquets de Python instal·lats:
- `matplotlib`
- `numpy`
- `time`
- `torchvision`
- `torch`
- `os`
- `sys`
- `skimage.color`
- `splitfolders`

En cas de no tenir instal·lats `pytorch` i `splitfolders` el codi per descarregar els paquets és el següent:
```
pip install pytorch splitfolders
```
Per clonar el directori en local, dirigiu-vos al directori on es vulgui guardar i escriviu la següent comanda:
```
git clone https://github.com/DCC-UAB/xnap-project-matcad_grup_1.git
```

### Carregar les imatges
Per fer aquest treball ens hem basat en 3 datasets d'imatges, en cas de voler tractar amb models ja entrenats haureu d'anar als següents links, descarregar-vos el dataset d'imatges i carregar-lo dins de la carpeta `data`.

- [Random](https://uofi.app.box.com/s/1cpolrtkckn4hxr1zhmfg0ln9veo6jpl)
- [Food](https://drive.google.com/file/d/1k_UvYzdrHbphW4UcbDb9jWB0ZQIAGEAo/view) 
- [Faces](https://github.com/2014mchidamb/DeepColorization/tree/master/face_images)

## Estructura del codi

El projecte està organitzat en diferents directoris que contenen codi executable, paquets de Python amb diferents classes i funcions i el dataset amb el conjunt d'imatges.
> Estructura general del repositori

```
.
├── checkpoints      # models entrenats amb diferents datasets (format .pt)
├── data                  # conjunt de fotografies
├── models             # carpeta amb el model
├── utils                  # paquets de python i funcions
├── main.py       
├── test.py       
├── train.py
└── README.md
```

- **checkpoints:** s'hi guarden els models entrenats amb format `.pt`. S’hi guarden dos models per dataset, un que ens diu en quina època ha obtingut una menor loss i un arxiu del model un cop ha acabat l’entrenament. A més s’hi guarden els `.pth` que guarden també els estats de la loss i la època.
- **data:** es troben diferents directoris amb diferents tipus d'imatge. Un cop executat el main, es crearà un directori amb les imatges separades en train i test.
- **models:** és on tenim els diferents models guardats.
- **outputs:** es crearan dues carpetes una `gray` i l'altre `color`. Dins de `gray` tindrem les imatges de validació en l'escala de grisos i dins de `color` és on es guardaran les diferents imatges acolorides.
- **utils:** Es guardaran els diferents fitxers amb funcions i classes necessàries per executar el main.
- **main:** Fitxer executable on s’inicialitzen els paràmetres d’execució, crea les carpetes amb les imatges resultants, entrena el model i el guarda en format `.pth` i `.pt`.
- **test:** Fitxer amb una única funció que valida el model utilitzant un conjunt de dades de validació. Mesura el temps i la *loss* durant la validació i guarda les imatges si s'especifica. Retorna la mitjana de la *loss*.
- **train:** Entrena el model amb un conjunt de dades d'entrenament. Calcula la loss, actualitza els pesos i mostra els resultats de l'entrenament.

Dins de la carpeta *Utils* tenim diferents arxius entre els quals podem trobar:
> Estructura de la carpeta utils
```
├── ...
├── utils                 
│   ├── meters       
│   ├── create_result_folders       
│   ├── convert_2_grayscale       
│   ├── split_data      
│   └── to_rgb              
└── ...
```

- **meters:** Ens permet inicialitzar, actualitzar i reiniciar les diferents variables.
- **create_result_folders:** Crea carpetes per distribuir les imatges i on guardarem els models entrenats.
- **split_data:** Dividim les dades entre train i val.
- **convert_2_grayscale:** Aquesta funció ens permet passar les imatges a blanc i negre.
- **to_rgb:** Ens permet passar les imatges a escala RGB.

A més, tenim la carpeta models amb el fitxer:
- **models:** Aquí tenim la definició del model utilitzat en el train.

## Execució del programa

### Executar models per entrenar
>Si dins la carpeta `data` es té **només una carpeta** amb les imatges simplement fer: `python main.py`.

>Si dins la carpeta `data` hi ha **més d'una carpeta amb diferents imatges** seguir les instruccions següents:
>1. Anar al fitxer `main.py`
>2. Canviar a la línia 28 la variable *folder_name* per '<nom_directori>', on <nom_directori> és el nom del directori amb les imatges amb les quals es vol entrenar el model.
>3. Només falta executar el main fent:`python main.py`.

**Exemple: volem entrenar el model amb les imatges de *random_images* i tenim la següent estructura amb tres directoris diferents:**

```
├── ...
├── data                 
│   ├── food_images      
│   ├── faces_images   
│   └── random_images              
└── ...
```
Anem al fitxer `main.py` i canviem la línia 28. En aquest cas la línia sencera quedaria: `images_folder = 'random_images'`.

Un cop acabat s'ha entrenat el model es guardarà dins la carpeta `checkpoints`.
### Executar models ja entrenats 
Per executar un model ja entrenat seguir les següents instruccions:
1. Anar a l'arxiu `main.py`
2. Descomentar les últimes línes de codi (línies 116 a 125)
3. En la comanda de *load_state_dic* canviar el format del path `checkpoints/NOM_MODEL.pth` on NOM_MODEL és el nom del model que es vulgui carregar.

Tots els models segueixen l’estructura de `<model>-epoch-<#epoch>-losses-<valor_loss>.pth`, a més a més, si teniu qualsevol dubte podeu anar a la carpeta `checkpoints` i allà hi seran tots els models per consultar els seus noms.







## Webgrafia i adreces d’interès
Per tal de fer aquest projecte s’ha utilitzat les següents fonts:

1. Carlos Julio Pardo. "Deep Learning: Colorización de Imágenes". Marzo 29, 2019. Disponible a: [Enllaç](https://carlosjuliopardoblog.wordpress.com/2019/03/29/deep-learning-colorizacion-de-imagenes/)

2. Canal de YouTube. "Tutorial de colorización de imágenes con Deep Learning". Disponible a: [Enllaç](https://www.youtube.com/watch?v=eXSJ94ldsoc&feature=youtu.be)

3. Canal de YouTube. "Image Colorization with Deep Learning - Part 1". Disponible a: [Enllaç](https://youtu.be/_JOvupaUcjU)

4. Canal de YouTube. "Image Colorization with Deep Learning - Part 2". Disponible a: [Enllaç](https://youtu.be/a-_vIs1zoBc)

5. Repositori de GitHub. "Colorful Colorization". Disponible a: [Enllaç](https://github.com/Time0o/colorful-colorization)

6. Kaggle. "PyTorch Pix2Pix for Image Colorization". Disponible a: [Enllaç](https://www.kaggle.com/code/orkatz2/pytorch-pix-2-pix-for-image-colorization/notebook)

7. Paper de conferència. "Coloring Black and White Images with Conditional Generative Adversarial Networks". Disponible a: [Enllaç](https://ceur-ws.org/Vol-2485/paper47.pdf)


