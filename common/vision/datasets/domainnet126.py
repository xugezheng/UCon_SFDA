"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import os
from typing import Optional
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class DomainNet126(ImageList):
    """`DomainNet <http://ai.bu.edu/M3SDA/#dataset>`_ (cleaned version, recommended)

    See `Moment Matching for Multi-Source Domain Adaptation <https://arxiv.org/abs/1812.01754>`_ for details.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'c'``:clipart, \
            ``'i'``: infograph, ``'p'``: painting, ``'q'``: quickdraw, ``'r'``: real, ``'s'``: sketch
        split (str, optional): The dataset split, supports ``train``, or ``test``.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            clipart/
            infograph/
            painting/
            quickdraw/
            real/
            sketch/
            image_list/
                clipart.txt
                ...
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/90ecb35bbd374e5e8c41/?dl=1"),
        ("clipart", "clipart.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip"),
        ("infograph", "infograph.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip"),
        ("painting", "painting.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip"),
        ("quickdraw", "quickdraw.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip"),
        ("real", "real.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip"),
        ("sketch", "sketch.zip", "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"),
    ]
    image_list = {
        "c": "image_list_126/clipart_list.txt",
        "p": "image_list_126/painting_list.txt",
        "r": "image_list_126/real_list.txt",
        "s": "image_list_126/sketch_list.txt",
    }
    list_dict = {"image_list": image_list}
    
    # 126 provided by https://github.com/MattiaLitrico/Guiding-Pseudo-labels-with-Uncertainty-Estimation-for-Source-free-Unsupervised-Domain-Adaptation
    CLASSES = ['aircraft_carrier', 'alarm_clock', 'bee', 'shoe', 'skateboard', 'snake', 'speedboat', 'spider', 'squirrel', 'strawberry', 'streetlight', 'string_bean', 'submarine', 'bird', 'swan', 'table', 'teapot', 'teddy-bear', 'television', 'The_Eiffel_Tower', 'The_Great_Wall_of_China', 'tiger', 'toe', 'train', 'blackberry', 'truck', 'umbrella', 'vase', 'watermelon', 'whale', 'zebra', 'blueberry', 'bottlecap', 'broccoli', 'bus', 'butterfly', 'cactus', 'cake', 'ant', 'calculator', 'camel', 'camera', 'candle', 'cannon', 'canoe', 'carrot', 'castle', 'cat', 'ceiling_fan', 'anvil', 'cello', 'cell_phone', 'chair', 'chandelier', 'coffee_cup', 'compass', 'computer', 'cow', 'crab', 'crocodile', 'asparagus', 'cruise_ship', 'dog', 'dolphin', 'dragon', 'drums', 'duck', 'dumbbell', 'elephant', 'eyeglasses', 'feather', 'axe', 'fence', 'fish', 'flamingo', 'flower', 'foot', 'fork', 'frog', 'giraffe', 'goatee', 'grapes', 'banana', 'guitar', 'hammer', 'helicopter', 'helmet', 'horse', 'kangaroo', 'lantern', 'laptop', 'leaf', 'lion', 'basket', 'lipstick', 'lobster', 'microphone', 'monkey', 'mosquito', 'mouse', 'mug', 'mushroom', 'onion', 'panda', 'bathtub', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'pig', 'pillow', 'pineapple', 'potato', 'power_outlet', 'bear', 'purse', 'rabbit', 'raccoon', 'rhinoceros', 'rifle', 'saxophone', 'screwdriver', 'sea_turtle', 'see_saw', 'sheep']

    def __init__(
        self, root: str, task: str, r: float, download: Optional[bool] = False, list_name = "image_list", **kwargs
    ):
        self.selected_list = self.list_dict[list_name]
        assert task in self.selected_list
        data_list_file = os.path.join(root, self.selected_list[task])


        super(DomainNet126, self).__init__(root, DomainNet126.CLASSES, data_list_file=data_list_file, r=r, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())
