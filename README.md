# avatar_gen_customized

Human to Cartoon Avatar Generator

Paper: https://arxiv.org/pdf/1611.07004.pdf

Dataset: https://huggingface.co/datasets/bhadresh-savani/photo-to-cartoon

## Downloads

Dataset as ZIP (Images): https://drive.google.com/file/d/1E_lXIFjYkrqiKW3Wlyro8yVqE8ivrFLY/view?usp=sharing

Dataset Train scr_image (x): https://drive.google.com/file/d/1RV5F0nQFV5ePAEdLYZA3ZkBHwj_9N8FJ/view?usp=sharing

Dataset Train tar_image (y): https://drive.google.com/file/d/1-7iXO528Bzg2u-US0tj4qFs80uBKYz8A/view?usp=sharing

Model 16_epochs: https://drive.google.com/file/d/1C8le-h40-frs-2QK3rjcPGwe-lIGqAHi/view?usp=sharing

## Results-
Epoch 20/20
 150/152 Discriminator-1_Loss: 0.112 Discriminator-2_Loss: 0.160 GAN_Loss: 7.010

Sample-

![image](https://drive.google.com/file/d/1U-dfxxlSbgvpsdds9mBSD6ochqLQlaes/view?usp=sharing)



## Steps to train and test the model-

1. 1_dataset_maker.py to create dataset
2. 2_dataset_to_npy.py to convert dataset to npy
3. 3_train_gan_pix2pix.ipynb to train model
