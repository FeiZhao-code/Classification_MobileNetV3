# Classification_MobileNetV3
## 一、环境要求
 所需环境
* Anaconda3（建议使用）版本问题不大
* python 3.7
* pycharm (IDE)
* pytorch GPU版 1.9 建议用Anaconda3配置此环境，具体百度。在pycharm 中使用此环境如下

## 二、项目文件介绍
### Classification_MobileNetV3文件夹存放使用pytorch实现的py代码
**model_v3.py**： 是模型的定义文件，不用修改  
**train.py**： 是调用模型训练的文件，可修改超参数    
**predict.py**： 是调用模型进行预测的文件  
**class_indices.json**： 是训练数据集对应的标签文件，此文件是运行train.py自动生成的，不需要修改
**pth2pt.py**:是模型格式转换文件，负责将pth文件转换为能部署在移动端的pt文件
### model文件夹：存放预训练模型（小写），训练好的模型（大写pth文件）和转换后的pt文件
### image文件夹：该文件夹是用来存放训练样本的目录
使用步骤如下：
* （1）在此original文件夹为每个类别单独创建文件夹存放图片

* （2）执行"split_data.py"脚本自动将original数据集划分成训练集train和验证集val    
├── image文件夹   
       ├── original（数据集文件夹）  
       ├── train（生成的训练集）  
       └── val（生成的验证集） 
 
## 三、视频讲解

