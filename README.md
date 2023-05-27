[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/sPgOnVC9)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11118680&assignment_repo_type=AssignmentRepo)
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
**Per tal de poder entrenar un model amb un dataset particular és important seguir aquestes instruccions**: 

Primerament entrar al fitxer main i modificar les següents variables:
- **input_path:** Aneu al lloc on us heu descarregat el projecte, un cop allà obriu la carpeta del projecte, obriu la carpeta 'data' i dins de data escolliu la carpeta d'imatges amb la que voleu executar el treball. Un cop escollida, copieu el path i l'enganxeu en aquesta variable.
- **output_path:** En aquest cas només haureu de canviar 'food_images' del path pel nom de la carpeta que anteriorment heu escollit.

Més avall del codi en la part d'opcions haureu de modificar les variables train_path i val_path.
Per aquestes variables haureu d'anar al projecte obrir la carpeta data a continuació l'arxiu d'imatges que esteu executant, obrir split_images i depenent si és per la variable train o val obrir cada carpeta seva respectiva copiar el path a cada variable corresponent.

Per ultim modifique la variable model_save_path per tal de posar el nom de com vodreu guardar el model importat un cop entrenat.

Amb això ja estarà tot llest per executar i entrenar el model. Per fer-ho escrivim al terminal:
```
python main.py

```
Un cop acabat el model d’entrenat es guardarà dins la carpeta checkpoints.

### Executar models ja entrenats 


Haureu de descomentar les últimes línies de codi i en la comanda de load_state_dic canviar el format del path 'checkpoints/NOM_MOEL.pth' i posar en comptes de NOM_MODEL posar el nom del model que voleu carregar. Per mirar el nom dels models haureu d'anar a la carpeta checkpoints i escollir-ne un.
Per tal de facilitar l’entendiment dels noms tots els models segueixen l’estructura de Nom_imatges_entrenament-epoch-Nom_de_epoch-losses-Valor_loss.pth 







## Webgrafia i adreces d’interès
Per tal de fer aquest projecte s’ha utilitzat les següents fonts:

1. Carlos Julio Pardo. "Deep Learning: Colorización de Imágenes". Marzo 29, 2019. Disponible a: [Enllaç](https://carlosjuliopardoblog.wordpress.com/2019/03/29/deep-learning-colorizacion-de-imagenes/)

2. Canal de YouTube. "Tutorial de colorización de imágenes con Deep Learning". Disponible a: [Enllaç](https://www.youtube.com/watch?v=eXSJ94ldsoc&feature=youtu.be)

3. Canal de YouTube. "Image Colorization with Deep Learning - Part 1". Disponible a: [Enllaç](https://youtu.be/_JOvupaUcjU)

4. Canal de YouTube. "Image Colorization with Deep Learning - Part 2". Disponible a: [Enllaç](https://youtu.be/a-_vIs1zoBc)

5. Repositori de GitHub. "Colorful Colorization". Disponible a: [Enllaç](https://github.com/Time0o/colorful-colorization)

6. Kaggle. "PyTorch Pix2Pix for Image Colorization". Disponible a: [Enllaç](https://www.kaggle.com/code/orkatz2/pytorch-pix-2-pix-for-image-colorization/notebook)

7. Paper de conferència. "Coloring Black and White Images with Conditional Generative Adversarial Networks". Disponible a: [Enllaç](https://ceur-ws.org/Vol-2485/paper47.pdf)


