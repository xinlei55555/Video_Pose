# VideoMambaPose
### Project combining VideoMamba model and ViTPose Heatmap Joint Regressor for Video Pose detection

## Installation:
#### clone the VideoMamba repository
```bash
git clone https://github.com/OpenGVLab/VideoMamba
cd VideoMamba
```
#### create your environment
```bash
conda create -n mamba python=3.10
conda activate mamba
pip install -r requirements.txt
pip install -e causal-conv1d
pip install -e mamba
```
#### clone the VideoMambaPose repository
```bash
cd ..
git clone https://github.com/xinlei55555/Video_Pose.git
pip install -r requirements125.txt
```

