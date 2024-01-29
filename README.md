# ImageClassification
Image Classification Project from Giordano Vitale &amp; Jan Philip Richter



Convolutional neural networks are great machine learning models to deal with the task of
image classification. This report introduces multiple model specifications in order to compare
their performance on a classification task with the goal of identifying whether an image contains
a muffin or a chihuahua.
We propose several different architectural approaches and evaluate their benefits and draw-
backs. We first consider two standard sequential architectures, varying in depth and com-
plexity. Moreover, hyperparameter tuning is performed to further optimise one of the models’
training phases. The second family of models adopts the VGG framework with the aim to
compare whether this already established architecture yields improvements over our previous
models. Lastly, a pair of residual neural networks is built to investigate the effect of a vastly
deeper and more complex model architecture. All models are compared in predictive accuracy
on the training- and validation data. One of the proposed models is selected to compute a
more accurate risk estimate via cross-validation.
Additionally, we perform transfer learning to analyse how well the classification performance
can be improved when considering pre-trained, state-of-the-art neural networks.
Finally, we discuss our obtained results and propose possible improvements.
