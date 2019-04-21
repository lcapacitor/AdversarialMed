# AdversarialMed: Adversarial Attacks and Defenses on Medical Imaging Deep Neural Networks

This repository is for the course project of CSC2516 Neural Networks and Deep Learning (2019). 

![AdversarialMed](https://github.com/lcapacitor/AdversarialMed/blob/master/attack_defense/figures/adv_example3_.png)

In this project, we investigated the influence of different mainstream adversarial attacks (both white-and black-box including FGSM, I-FGSM, PGD, and MI-FGSM), on a pneumonia diagnosis DNN (namely PneuNet) which is transfer-learned from [CheXNet] with the data from [Kaggle competition]. We found that the PneuNet model is highly susceptible to white-box attacks. By applying the FGSM, the test accuracy and AUC score of PneuNet will plunge from 0.9547 and 0.9910 down to 0.0703 and 0.0047 respectively, while both metrics will drop down to 0 when applying any iterative based attacks.

Moreover, we further analyzed the performance of different type of defense approaches, which includes JPEG compression, adversarial learning, and VAE denoising. We suggest that the adversarial learning provides our model with the best protection from various adversarial attacks (AUC>0.95) while maintains a relative high performance (AUC≈0.97) on the clean data at the same time.

We would suggest all practitioners in the machine learning for health community could take adversarial attacks into account when constructing their DNN models since the benefits of the adversarial learning would be twofold:

  - Involving with adversarial loss in training, even with a one-step adversarial method (FGSM), would significantly improve the robustness against various types of attacks.
  - The adversarial loss provides extra regularization power to the training process that would be beneficial in preventing overfitting, especially when the tasks don’t have sufficient amount of data for the model to learn from.

# Authors

  - Yan (Leo) Li, Department of Electrical and Computer Engineering
  - Yuyang (Nathan) Liu, Department of Computer Science

License
----

MIT

   [CheXNet]: <https://stanfordmlgroup.github.io/projects/chexnet/>
   [Kaggle competition]: <https://www.kaggle.com/goelrajat/prediciting-pneumonia-from-chest-xray/data>

