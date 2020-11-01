# Tutorial for Object detection using Region Proposal Network
by Dufan Wu: dwu6@mgh.harvard.edu
Center for Advanced Medical Computing and Analysis, Massachusetts General Hospital, Boston MA 02114
https://camca.mgh.harvard.edu/

This is a tutorial using region proposal network to detect lung in chest X-ray. The ground truth bounding box is generated from the lung segmentation.<br>
The dataset is from https://www.kaggle.com/nikhilpandey360/chest-xray-masks-and-labels

Data source: National Library of Medicine, National Institutes of Health, Bethesda, MD, USA and Shenzhen No.3 People’s Hospital, Guangdong Medical College, Shenzhen, China<br>
Publications on this data:<br>
[1] Jaeger S, et al. Automatic tuberculosis screening using chest radiographs. IEEE Trans Med Imaging. 2014 Feb;33(2):233-45.<br>
[2] Candemir S, et al. Lung segmentation in chest radiographs using anatomical atlases with nonrigid registration. IEEE Trans Med Imaging. 2014 Feb;33(2):577-90.

The Region Proposal Network (RPN) is the backbone for modern object detection networks. It outputs a list of bounding boxes, with the probability of an object exists in each bounding box. Compared to classification, detection gives the location of objects; compared to segmentation, detection only needs very simple label (bounding boxes) to train, rather than a semantic segmentation.

## Steps to use the tutorial
### Step 1: Install conda environment
Run `conda env create -f lung_detection_tutorial.yml` and `conda activate lung_detection_tutorial`

### Step 2: Download dataset
Download dataset from https://www.kaggle.com/nikhilpandey360/chest-xray-masks-and-labels. You may need a kaggle account to do this. Unzip the files to `path\to\your\directory`.

### Step 3: Preprocess the dataset
Run through the `preprocess.ipynb` to generate input files from the original dataset. It will zoom the images to smaller size, get bounding boxes from the segmentation and put everything in the `data` folder. Change this line to the path you have unzipped the file:
```python
dataset_path = '/raid/temp/Lung Segmentation/'
```

### Step 4: Training and testing
Run through the `RPN.ipynb` for the RPN label preparing, training, and testing. You can play with the parameters and see how it goes. There is already a pretrained model at `train/model.h5` at 100 epochs. However, if you want to train from scratch, comment this line out:
```python
check_point = './train/model.h5'   # comment this line out if you want to train from scratch
```
The training takes 20 to 30 minutes on a V100. 

If you want see how the model works during testing without training, you can skip the training block or set nepochs to 0.