As mentioned in our paper, we develop our project based on Su et al.,’s [1] implementation of MVCNN (we use codes from [here](https://github.com/jongchyisu/mvcnn_pytorch)). We implement our work with Python3.5.2 and Pytorch1.0.1.post2.  
We conduct the experiments on a single Tesla K80 GPU with CUDA9.0.  
For the sake of GPU memory consumption, we employ a two stage training strategy.We utilize a pretrained VGG-M model for the feature extraction, and train our VMM model with the features as input. To run our codes, you should  
## – Install the required dependencies, including:
>pytorch  
>tensorboardX  
>numpy  
>scipy  
>For the detailed version, you can refer to the “requirement.txt”. (Note: not all the libs listed in requirement.txt are needed.)  
## – Prepare the dataset:
>1. Download dataset Modelnet40 from [here](http://modelnet.cs.princeton.edu/)
>2. Render 2D images by blender (we use codes from [here](https://github.com/jongchyisu/mvcnn_pytorch))
>3. Save the images, and arrange the dataset directory as follow:
>>/MVCNN dataset/class folders/train(or test)/rendered images
## – Prepare the feature extractor(VGG-M) and extract the features:
>1. Clone codes from xxx and put it under the root directory
>2. We utilize a [VGG-M pretrained on imagenet] (http://data.lip6.fr/cadene/pretrainedmodels/vggm-786f2434.pth) and than finetune it on ModelNet40 as the feature extractor.
>We adopt Su’s MVCNN codes to finetune the model.Due to the limitation of the size to the uploaded file, we cannot provide a pre-trained VGGM model in the supplementary materials directly. Therefore, we introduce the way to trained VGG-M on ModelNet40 instead:
>>`cd feature extractor`  
>>`python train mvcnn.py -name xxx -cnn_name vggm -train_path xxx -val_path xxx`  
>For example:  
>>`python train mvcnn.py -name MVCNN -cnn_name vggm -train_path /Modelnet40_dataset_rendered_images/*/train -val_path /Modelnet40_dataset_rendered_images/*/test`  
>3. Extract features with the trained vgg-m:
>>`python mvcnn_save_feature.py -name xxx -cnn_name vggm`  
>You should change the path of your pretrained model and the save direction in mvcnn_save_feature.py. Then you can get the saved features.
## – To train our VMM on ModelNet40 with default settings:
>1. Clone this repo into the root directory and named as VMM
>2. Specify the path where the extracted features are saved, and run:
>>`cd VMM`  
>>`CUDA VISIEBLE DEVICES=0 python train nem.py -name NEM -train_pth xxx -val_path xxx`  
>Then, you can check the training process with tensorboard. The log file as well as the trained model can be find at VMM/exp/NEM
## - To try different settings, you can:
>specify the number of latent views, for example, set it to 3 by “-cluster n 3”;
>specify the number of iteration, for example set it to 10 by “-iter 10” ;
>for more settings, you can refer to VMM/train nem.py

References  
`1. Su, J.C., Gadelha, M., Wang, R., Maji, S.: A deeper look at 3d shape classifiers. In The European Conference on Computer Vision (ECCV) Workshops (September 2018)`
