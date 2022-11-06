# TCS-LBCNN
Torch implementation of - Threshold Center-Symmetric Local Binary Convolutional Neural Networks for Bilingual Handwritten Digit Recognition

## Abstract
The writing style of the same writer varies from instance to instance in Arabic and English handwritten digit recognition, making handwritten digit recognition challenging. Currently, deep learning approaches are applied in many applications, including convolutional neural networks (CNNs) modified to produce other models, such as local binary convolutional neural networks (LBCNNs). An LBCNN is created by fusing a local binary pattern (LBP) with a CNN by reformulating the LBP as a convolution layer called a local binary convolution (LBC). However, LBCNNs suffer from the random assignment of 1, 0, or -1 to LBC weights, making LBCNNs less robust. Nevertheless, using another LBP-based technique, such as center-symmetric local binary patterns (CS-LBPs), can address such issues. In this paper, a new model based on CS-LBPs is proposed called center-symmetric local binary convolutional neural networks (CS-LBCNN), which addresses the issues of LBCNNs. Furthermore, an enhanced version of CS-LBCNNs called threshold center-symmetric local binary convolutional neural networks (TCS-LBCNNs) is proposed, which addresses another issue related to the zero-thresholding function. Finally, the proposed models are compared to state-of-the-art models, proving their ability by producing a more accurate and significant classification rate than the existing LBCNN models. For the bilingual dataset, the TCS-LBCNN enhances the accuracy of the LBCNN and CS-LBCNN by 0.15% and 0.03%, respectively. In addition, the comparison shows that the accuracy acquired by the TCS-LBCNN is the second-highest using the MNIST and MADBase datasets.

***
### Paper Download
https://doi.org/10.1016/j.knosys.2022.110079

## Research Aims and Objectives
This research aims to enhance the performance of LBP-based convolutional neural networks on the automatic recognition of bilingual handwriting. The objectives of the research are as follows:
i.	To introduce center-symmetric local binary convolutional neural networks (CS-LBCNNs) to overcome the illumination transformation and the negative effect of the random weights of LBCNNs.
ii.	To enhance the CS-LBCNN by applying a nonzero thresholding function that allows the model to extract more distinguished features, called the threshold center-symmetric local binary convolutional neural network (TCS-LBCNN) model.
iii.	To validate the models with other benchmark models.

***

## References

* Al-wajih, Ebrahim, and Rozaida Ghazali. "Threshold center-symmetric local binary convolutional neural networks for bilingual handwritten digit recognition." Knowledge-Based Systems (2022): 110079.

```
@article{al2022threshold,
  title={Threshold center-symmetric local binary convolutional neural networks for bilingual handwritten digit recognition},
  author={Al-wajih, Ebrahim and Ghazali, Rozaida},
  journal={Knowledge-Based Systems},
  pages={110079},
  year={2022},
  publisher={Elsevier}
}
```


***

### Requirements
See the [installation instructions](INSTALL.md) for a step-by-step guide.



