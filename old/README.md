# PatchCore anomaly detection
Unofficial implementation of PatchCore(new SOTA) anomaly detection model


Original Paper : 
Towards Total Recall in Industrial Anomaly Detection (Jun 2021)  
Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, Peter Gehler  


https://arxiv.org/abs/2106.08265

notice(21/06/18) :  
This code is not yet verified. Any feedback is appreciated.  

### Usage 
~~~
# python 3.6
pip install -r requirements.txt
python train.py --phase train or test --dataset_path .../mvtec_anomaly_detection --category carpet --project_root_path path/to/save/results --coreset_sampling_ratio 0.01 --n_neighbors 9'
~~~

### MVTecAD AUROC score (PatchCore-1%, mean of n trials)
| Category | Paper<br>(image-level) | This code<br>(image-level) | Paper<br>(pixel-level) | This code<br>(pixel-level) |
| :-----: | :-: | :-: | :-: | :-: |
| carpet | 0.980 | 0.995(1) | 0.989 | 0.989(1) |
| grid | 0.986 | 0.899(1) | 0.986 | 0.978(1) |
| leather | 1.000 | 1.000 | 0.993 | 0.992(1) |
| tile | 0.994 | 0.981(1) | 0.961 | 0.932(1) |
| wood | 0.992 | - | 0.951 | - |
| bottle | 1.000 | - | 0.985 | - |
| cable | 0.993 | - | 0.982 | - |
| capsule | 0.980 | - | 0.988 | - |
| hazelnut | 1.000 | - | 0.986 | - |
| metal nut | 0.997 | - | 0.984 | - |
| pill | 0.970 | - | 0.971 | - |
| screw | 0.964 | - | 0.992 | - |
| toothbrush | 1.000 | - | 0.985 | - |
| transistor | 0.999 | -| 0.949 | - |
| zipper | 0.992 | - | 0.988 | - |
| mean | 0.990 | - | 0.980 | - |

### Code Reference
https://github.com/google/active-learning  
https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
