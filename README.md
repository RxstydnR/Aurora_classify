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

## main.py

This is main code for classifying auroral images using a deep learning model.

- **`model.py`** has some kinds of deep learning model.

- **`data.py`** has data handling methods to prepare data for training and evaluation.


Run `"main.py"`

```bash
python main.py --data_path [Image folder] --save_path [Save folder]
```

If you don't use .npy format data, set the Image folder which has the same structure as "SampleDataset" above. <br>

You can see the details of the other arguments in the code.

---

## prediction.py

This is for classifying auroral images using **trained** model.

Run `"prediction.py"`

```bash
python prediction.py --data_path [Image folder] --save_path [Save folder] --fps 20.0
```

If you don't use .npy format data, set the Image folder which has the same structure as "SampleDataset" above. <br>

---

## hsv_histgram_classifier.py

This program classify aurora and cloud images using **HSV color space** with reference to [1].<br>
You cannot run this program, the only use is to refer to the methods in it.

---

## Note

**I test environments under Mac, not Windows.**

## References

[1] 田中孝宗, et al. "オーロラの出現・形状の予測に向けた全天オーロラ画像の自動分類への試み." 宇宙航空研究開発機構研究開発報告: 宇宙科学情報解析論文誌: 第 4 号 (2015): 127-134.
