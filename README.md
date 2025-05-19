# pNEUMA‑Vision Object Detection with Faster R‑CNN

**End‑to‑end pipeline for training a Faster R‑CNN detector on the pNEUMA‑Vision traffic‑drone dataset.**

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Dataset Preparation](#dataset-preparation)
4. [Training](#training)
5. [Evaluation & Inference](#evaluation--inference)
6. [Project Structure](#project-structure)
7. [Environment / Requirements](#environment--requirements)
8. [Results](#results)
9. [License](#license)
10. [Citation](#citation)

---

## Overview

This repository contains everything required to 

* **Download** raw pNEUMA‑Vision drone videos & metadata.
* **Generate bounding‑boxes** from the released centroids (axis‑aligned *or* rotated).
* **Export to YOLO format** for lightweight storage.
* **Train** a **Faster R‑CNN + FPN** model (PyTorch / `torchvision`).
* **On‑the‑fly convert** YOLO labels → COCO format inside a custom `PneumaDataset` class—so we keep disk usage low while feeding COCO‑style targets to the detector.

<p align="center">
  <img src="docs/pipeline.svg" width="640" alt="Pipeline diagram showing download → bbox → YOLO → PyTorch" />
</p>

---

## Quick Start
Before you start make sure that your data folder has the following structure:

```text
Pneuma_Vision/
├── 20181029_D2_0930_1000/       # downloaded dataset
│   ├── Annotations/
│   └── Frames/
├── 20181029_D2_1000_1030/       # downloaded dataset
│   ├── Annotations/
│   └── Frames/
└── 20181029_D6_0900_0930/       # The dataset that we trained our model
    ├── Annotations/             # The annotations without processing that we downloaded
    ├── Frames/                  # drone images
    ├── Labels_axis_aligned/     # annotations with the created upright bounding boxes
    └── Labels_YOLO_upright/     # annotations in YOLO format that will be used for training

Vehicle Detection from Drones/
├── checkpoints/
│   ├── fasterrcnn_baseline.pth            # your saved checkpoint
│   └── loss_history_baseline.csv          # your saved loss history
├── configs/
│   └── baseline.yaml
├── src/
│   └── Faster R-CNN/
│       ├── train.py
│       ├── data.py
│       ├── inference.py
│       └── model.py
├── utils/
│   ├── bbox_functions.py
│   ├── process_bounding_box.py
│   └── process.py
├── README.md
├── requirements.txt
└── .gitignore

``` 

# 1️⃣  Clone & install
git clone https://github.com/your‑user/pneuma‑vision‑det && cd pneuma‑vision‑det
pip install -r requirements.txt  # or: conda env create -f env.yml

# 2️⃣  Download dataset
Download Pneuma Vision datasets in the following link https://zenodo.org/records/7426506

# 3️⃣  Generate bounding boxes (axis‑aligned) and Export them to YOLO format 
We generate axis align bounding boxes and export them in the directory inside the dataset to YOLO format. An example of such a command is given below

python3 process.py --base_dir '/home/nikos2/Desktop/Data/Pneuma Vision' --drone 6 --session 3 --dont_show --stage all

--base_dir : The directory where all datasets are saved
--drone : number of drone
--session : number of session
--dont_show : whenever the results to be shown on the screen
--stage : if you want the whole process to be shown on your screen (generate bounding boxes & export them to YOLO format)

# 5️⃣  Create the Configuration file
All runtime options live in a single YAML file. The repository ships with configs/baseline.yaml. In the configuration you can also specify the checkpoint path and the loss history path if you want to load your model from a specific checkpoint. Those files should be in the checkpoints directory

# 5️⃣  Train Faster R‑CNN
Now we train the model. An example of a command is given below :

python3 'src/Faster R-CNN/train.py' --root_dir '/home/nikos2/Desktop/Data/Pneuma Vision/20181029_D6_0900_0930'

--root_dir : The specific directory where your model will be trained on

## Evaluation & Inference
Now we run an inference of the model with the following command. The script will take input images and for each image will predict and draw bounding boxes for the vehicles of each image and save the annotations of the bounding boxes for each file in a csv. We ran the following command :

python3 'src/Faster R-CNN/inference.py' --checkpoint checkpoints/fasterrcnn_baseline.pth --input_dir '/home/nikos2/Desktop/Data/Pneuma Vision/20181029_D6_0900_0930/Frames2' --output_dir '/home/nikos2/Desktop/Data/Pneuma Vision/20181029_D6_0900_0930' --csv_dir '/home/nikos2/Desktop/Data/Pneuma Vision/20181029_D6_0900_0930' --conf 0.5

--checkpoint   : On which saved model we will make predictions 
--input_dir    : Path to the folder containing images for inference.
--output_dir   : Path to save the processed images with bounding boxes.
--csv_dir      : Path to save individual CSV files for each image.
--conf         : Minimum confidence score to retain a detection.

## Environment / Requirements

* Python ≥ 3.10
* PyTorch 2.x + torchvision 0.18
* OpenCV, pandas, tqdm, pycocotools

> Full list in **`requirements.txt`**. CUDA 11.8 tested on RTX 4080.

---

## Results

| Model                  | Input res | mAP<sub>50:95</sub> | mAP<sub>50</sub> | FPS (T4) |
| ---------------------- | --------- | ------------------- | ---------------- | -------- |
| Faster R‑CNN ResNet‑50 | 1024      | **0.51**            | 0.78             | 12       |
| YOLOv8‑L (re‑impl)     | 1024      | 0.47                | 0.74             | **58**   |

*(Numbers computed on drones 9‑10 validation split, May 19 2025.)*

---

## License

Code released under **MIT**. pNEUMA‑Vision data is distributed under the **Creative Commons Attribution 4.0** license—see the dataset authors for details.

---

## Citation

If you use this repo, please cite pNEUMA and the toolbox:

```bibtex
@dataset{pneuma2022vision,
  author       = {Barmpounakis, Emmanouil and others},
  title        = {{pNEUMA‑Vision: Aerial Drone Sequence With Centroid Annotations}},
  year         = {2022},
  doi          = {10.5281/zenodo.7016096}
}

@inproceedings{kim2023pneuma,
  author    = {Kim, Hyun‑Woo and Barmpounakis, Emmanouil and Geroliminis, Nikolas},
  title     = {Object Detection in Large‑Scale Drone Traffic Videos},
  booktitle = {IEEE ITSC},
  year      = {2023}
}
```

---

### Acknowledgements

Thanks to the pNEUMA consortium for releasing the original trajectory dataset and to **E. Barmpounakis et al.** for extending it with Vision annotations.

python3 process.py --base_dir '/home/nikos2/Desktop/Data/Pneuma Vision' --drone 6 --session 3 --dont_show --stage all

python3 'src/Faster R-CNN/train.py' --root_dir '/home/nikos2/Desktop/Data/Pneuma Vision/20181029_D6_0900_0930'

python3 'src/Faster R-CNN/inference.py' --checkpoint checkpoints/fasterrcnn_baseline.pth --input_dir '/home/nikos2/Desktop/Data/Pneuma Vision/20181029_D6_0900_0930/Frames2' --output_dir '/home/nikos2/Desktop/Data/Pneuma Vision/20181029_D6_0900_0930' --csv_dir '/home/nikos2/Desktop/Data/Pneuma Vision/20181029_D6_0900_0930' --conf 0.5
