This work is aimed at automatically tuning a pre-trained CNN layers with the knowledge of target dataset for improved transfer learning. 
Abstract:
Transfer learning enables solving a specific task having limited data by using the pre-trained deep networks trained on large-scale datasets. Typically, while transferring the learned knowledge from source task to the target task, the last few layers are fine-tuned (re-trained) over the target dataset. However, these layers are originally designed for the source task that might not be suitable for the target task. In this paper, we introduce a mechanism for automatically tuning the Convolutional Neural Networks (CNN) for improved transfer learning. The pre-trained CNN layers are tuned with the knowledge from target data using Bayesian Optimization. First, we train the final layer of the base CNN model by replacing the number of neurons in the softmax layer with the number of classes involved in the target task. Next, the CNN is tuned automatically by observing the classification performance on the validation data (greedy criteria). To evaluate the performance of the proposed method, experiments are conducted on three benchmark datasets, e.g., CalTech-101, CalTech-256, and Stanford Dogs. The classification results obtained through the proposed AutoTune method outperforms the standard baseline transfer learning methods over the three datasets by achieving 95.92%, 86.54%, and 84.67% accuracy over CalTech-101, CalTech-256, and Stanford Dogs, respectively. The experimental results obtained in this study depict that tuning of the pre-trained CNN layers with the knowledge from the target dataset confesses better transfer learning ability.

![Alt Text](Motivation_AutoTune1.png?raw=true "Title")
Fig. Overview of the proposed method. a) Typically, the CNN models used for transferring the knowledge are initially trained over a large-scale image dataset. b) Conventionally, while transferring the learned knowledge from the source task to the target task, the last one or a few layers of the pre-trained CNN are fine-tuned over the target dataset. c) The proposed AutoTune method tunes the $k$ number of layers automatically using Bayesian Optimization \cite{frazier2018tutorial}. Note that the lock and unlock symbols are used to represent the frozen and fine-tuned layers, respectively. Finally, the tuned CNN layers can be re-trained over the target dataset for improved transfer learning. Different colors represent different CNN layers.



Citation
@article{basha2020autotune,
  title={AutoTune: Automatically Tuning Convolutional Neural Networks for Improved Transfer Learning},
  author={Basha, SH and Vinakota, Sravan Kumar and Pulabaigari, Viswanath and Mukherjee, Snehasis and Dubey, Shiv Ram},
  journal={arXiv preprint arXiv:2005.02165},
  year={2020}
}
