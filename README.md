<div align="center" style="text-decoration: none !important;">
    <h1>
      <a href="https://ieeexplore.ieee.org/document/11202498" target="_blank" style="text-decoration: none !important;">[TGRS'2025] Adaptive Expert Learning for Hyperspectral and Multispectral Image Fusion</a> 
    </h1>
    <div>
        <a href='https://github.com/Hewq77' target='_blank' style="text-decoration: none !important;">Wangquan He<sup></sup></a>,&emsp;
        <a href='#' target='_blank' style="text-decoration: none !important;">Yixun Cai<sup></sup></a>,&emsp;
        <a href='https://github.com/renqi1998' target='_blank' style="text-decoration: none !important;">Qi Ren<sup></sup></a>,&emsp;
        <a href='#' target='_blank' style="text-decoration: none !important;">Abuduwaili Ruze<sup></sup></a>,&emsp;
        <a href='https://scholar.google.com/citations?user=UxbDMKoAAAAJ&hl=zh-CN&oi=ao' target='_blank' style="text-decoration: none !important;">Sen Jia<sup>&#8224;</sup></a>
    </div>
    <div>
        <sup></sup>College of Computer Science and Software Engineering, Shenzhen University &emsp;
    </div>
</div>

## ⚙️ Environment
```
conda create -n aelf python=3.9
conda activate aelf
pip install -r requirements.txt
```
## 🛫 Usage
1.  **Datasets setting**
   - Download  Datasets : [Pavia](https://ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes) / [DFC 2018](http://hyperspectral.ee.uh.edu/QZ23es1aMPH/2018IEEE/phase2.zip) / [Chikusei](https://naotoyokoya.com/Download.html).

- Put it in the `./Datasets`.

2.  **Training**
 ```
 CUDA_VISIBLE_DEVICES=0 
 python main.py  \
    -arch 'AELF' \
    -dataset 'Pavia' \
    --scale_ratio 4 \
    --model_path './checkpoints'\
```

3.  **Inference**
```
python test.py
```

## 🎓 Citations
Please cite us if our work is useful for your research.
```
@ARTICLE{He2025AELF,
  author={He, Wangquan and Cai, Yixun and Ren, Qi and Ruze, Abuduwaili and Jia, Sen},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Adaptive Expert Learning for Hyperspectral and Multispectral Image Fusion}, 
  year={2025},
  volume={63},
  number={},
  pages={1-15},
publisher={IEEE}}
