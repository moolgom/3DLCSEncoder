# 3D TSDF Volume Encoder

![representative2023](representative2023.jpeg)

This repository implements [TSDF Volume Compression with Axis-wise Variable Resolution Representation and Selective Latent Code Encoding].

***(Due to intellectual property issues related to this research, all code will be released upon the official publication of the paper.)***


# Abstract

This paper presents a novel approach for compressing truncated signed distance function (TSDF) volumes, leveraging two key contributions: an axis-wise variable resolution representation and a latent code selection-based compression model. The axis-wise variable resolution representation adapts the resolution along each axis based on local geometric complexity, effectively reducing the data size while preserving intricate geometric details. The proposed compression model employs optimal latent code selection, leading to improved compression efficiency and reduced computational complexity. The combination of these contributions results in a synergistic effect, enabling enhanced compression efficiency and preservation of geometric details in high-resolution TSDF volumes. 


# Encoder Performance
<!--
![](./Images/dragon.png =250x250) 
![](./Images/dragon.png =250x250)


<table>
<tr>
<td>
<img src="./Images/dragon.png" width="200" height="120">
  Uncompressed
</td>
<td>
<img src="./Images/dragon_draco_QP10.png" width="200" height="120">
  Draco QP=10
</td>
<td>
<img src="./Images/dragon_ours_RP8.png" width="200" height="120">
  Ours RP=8
</td>
</tr>
</table>
-->

<img src="./Images/dragon.png" width="500" height="300">
Uncompressed

<img src="./Images/dragon_draco_QP10.png" width="500" height="300">
Draco QP=10 (489.5342 KB)

<img src="./Images/dragon_ours_RP8.png" width="500" height="300">
Ours RP=8 (470.9717 KB)

Dragon (http://graphics.stanford.edu/data/3Dscanrep/)               
| QP | KB       | Chamfer Dist. | Rate Point | KB       | Chamfer Dist. |
|----|----------|---------------|------------|----------|---------------|
| 8  | 314.7529 | 0.000662      | 0          | 151.8047 | 0.000226      |
| 9  | 388.373  | 0.00037       | 1          | 154.5332 | 0.000216      |
| 10 | 489.5342 | 0.000189      | 2          | 155.8838 | 0.000199      |
| 11 | 607.5693 | 9.47E-05      | 3          | 160.9619 | 0.000184      |
| 12 | 702.7988 | 4.68E-05      | 4          | 172.5967 | 0.000159      |
| 13 | 824.1943 | 2.43E-05      | 5          | 200.4932 | 0.000135      |
| 14 | 944.4453 | 1.21E-05      | 6          | 233.582  | 0.000125      |
| 15 | 1059.284 | 5.96E-06      | 7          | 394.9395 | 0.000115      |
|    |          |               | 8          | 470.9717 | 8.24E-05      |
|    |          |               | 9          | 1020.904 | 7.79E-05      |
|    |          |               | 10         | 1969.086 | 7.78E-05      |
|    |          |               | 11         | 3295.279 | 7.47E-05      |




Thai Statuette (http://graphics.stanford.edu/data/3Dscanrep/)     
| QP | KB       | Chamfer Dist. | Rate Point | KB       | Chamfer Dist. |
|----|----------|---------------|------------|----------|---------------|
| 8  | 2107.906 | 1.128602      | 0          | 567.1553 | 0.220818      |
| 9  | 2644.877 | 0.586638      | 1          | 581.7754 | 0.213324      |
| 10 | 3420.002 | 0.315989      | 2          | 589.6455 | 0.201068      |
| 11 | 4298.582 | 1.74E-01      | 3          | 604.791  | 0.187856      |
| 12 | 4968.762 | 9.25E-02      | 4          | 656.043  | 0.1668        |
| 13 | 6311.539 | 4.64E-02      | 5          | 752.0625 | 0.146347      |
| 14 | 7942.594 | 2.32E-02      | 6          | 867.2246 | 0.125319      |
| 15 | 9585.34  | 1.16E-02      | 7          | 1473.012 | 0.124741      |
|    |          |               | 8          | 1762.874 | 8.97E-02      |
|    |          |               | 9          | 3519.432 | 8.86E-02      |
|    |          |               | 10         | 7194.209 | 8.78E-02      |
|    |          |               | 11         | 12655.88 | 8.50E-02      |

