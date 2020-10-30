# Deep Learning Classification of Aurora images

This code for classifying auroral images using a deep learning model, but can be used for other purpose as well.

# Usage
## make_npydata.py

The code is to make Image and label dataset as npy format.<br>
This makes loading image dataset faster when an experiment is conducted, but not necessaery.

As a default, a folder which has image dataset should be like the structure below.<br>
The dataset folder has each class folder, and also each of them contains the image data.

```bash
SampleDataset
├── class1
│   └── 1.jpg
├── class2
│   └── 2.jpg
├── class3
│   └── 3.jpg
├── class4
│   └── 4.jpg
└── class5
    └── 5.jpg
```

As a result, `data.npy` is created, its structure is like below.

```bash
data.npy
├── X image arrays
└── Y labels
```

Run `"make_npydata.py"`

```bash
python make_npydata.py --data_path [Image folder] --save_path [Save folder] --img_size [image size]
```

---

## make_movie.py

**In : images** <br>
**Out : a movie**

This code is for create a video from multiple image data using "opencv2".

Run `"make_movie.py"`

Note that image names should be consistent.(ex. 00001.jpg,00002.jpg) <br>
Unless the images in the video may not line up in the correct order.

```bash
python make_movie.py --data_path [Image folder] --save_path [Save folder] --fps 20.0
```

---

## opticalflow_image.py

**In : images** <br>
**Out : opticalflow images**

This code calculates the optical flow of image sequence and outputs it to image sequence.
**Optical flow parameters have to be adjusted by your hand.**


Run `"opticalflow_image.py"`

Note that image names should be consistent.(ex. 00001.jpg,00002.jpg) <br>
Unless the images in the video may not line up in the correct order.

```bash
python opticalflow_image.py --data_path [Image folder] --save_path [Save folder]
```

---

## opticalflow_movie.py

**In : movie** <br>
**Out : opticalflow movie**

This code calculates the optical flow of the video and outputs it to the video.
**Optical flow parameters have to be adjusted by your hand.**


Run `"opticalflow_movie.py"`

```bash
python opticalflow_movie.py --data_path [Movie path] --save_path [Save folder] --fps 20.0
```


## Note

MP4 video sometimes fails to be save.<br>
**.avi** format is used in these codes.

**I test environments under Mac, not Windows.**
