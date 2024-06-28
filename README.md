<p align="center">
  
  <h3 align="center"><strong>Clustering and Anomaly Detection in Embedding Spaces for the Validation of Automotive Sensors</strong></h3>

  <p align="center">
      <a href="https://www.linkedin.com/in/alexanderfertig/" target='_blank'>Alexander Fertig</a><sup>1</sup>&nbsp;&nbsp;
      <a href="https://www.linkedin.com/in/lakshman-balasubramanian-50548477/" target='_blank'>Lakshman Balasubramanian</a><sup>1</sup>&nbsp;&nbsp;
      <a href="https://www.thi.de/personen/prof-dr-ing-michael-botsch/" target='_blank'>Michael Botsch</a><sup>1</sup>&nbsp;&nbsp;
    <br>
    <small><sup>1</sup>Technische Hochschule Ingolstadt, Germany&nbsp;&nbsp;</small>
  </p>
</p>


> **Abstract:** In order to reliably validate autonomous driving functions, known risks must be taken into account and unknown risks must be identified. This work addresses this challenge by investigating risks at the level of object state estimations. The proposed methodology utilizes the differences between object state estimations from independent sensors, enabling the detection of relevant differences. This is a significant advantage, because sensor errors can be detected without ground truth. A deep autoencoder architecture is introduced to map the differences between state estimations into a latent space. The autoencoder contains Transformer and LSTM components to effectively process signals of varying lengths. The latent space is shaped using a k-means friendly design procedure, in order to find a suitable representation for anomaly detection. Detecting anomalies is a key component in the validation process of autonomous vehicles, contributing to the identification of unknown risks. The proposed approach is evaluated using real-world automotive sensor data from cameras and laser scanners in the publicly available nuScenes dataset. The results show that the generated latent space using the k-means friendly procedure is well suited for clustering differences between state estimations from these two sensors and thus for anomaly detection. In the framework specified in the safety standard ISO 21448 (SOTIF) the proposed methodology can play a key role for the detection of unknown risks on the perception level during the operation phase of autonomous vehicles.


<!-- omit in toc -->
## Overview
- [Installation](#installation)
- [Visualization](#visualization)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)



## Installation

### Clone this repository:

```bash
git clone https://github.com/MB-team-THI/sensor-difference-signal-embedding

.git
cd sensor-difference-signal-embedding


```

### Build the Docker image:

```bash
cd docker
docker build -t TBD -f Dockerfile .
```



## Visualization

Visualize the created latent embedding space by the interactive visualization tool [DIFFINSPECT](https://MB-Team-THI.github.io/DIFFINSPECT/), based on the [UMAP-Explorer](https://github.com/GrantCuster/umap-explorer/).

## Acknowledgments
The authors acknowledge the support of ZF Friedrichshafen AG.


We thank the authors of

- [StreamPETR](https://github.com/exiawsh/StreamPETR)
- [FocalFormer3D](https://github.com/NVlabs/FocalFormer3D)
- [UMAP-Explorer](https://github.com/GrantCuster/umap-explorer/)
  
for their open source contribution which made this project possible.

---



## Citation
```
@InProceedings{Fertig2024,
  author     = {Alexander Fertig and Lakshman Balasubramanian and Michael Botsch},
  booktitle  = {2024 IEEE Intelligent Vehicles Symposium (IV)},
  title      = {Clustering and Anomaly Detection in Embedding Spaces for the Validation of Automotive Sensors},
  year       = {2024},
  pages      = {1083-1090},
}
```
