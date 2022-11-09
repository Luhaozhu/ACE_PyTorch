from typing import List
from torch.utils.data import Dataset,DataLoader
import torch
from PIL import Image


        

class SingleConceptDataset(Dataset):
    """construct single concept activation dataset"""
    def __init__(self,activation):
        self.activation = torch.from_numpy(activation)
        
    def __getitem__(self, index):
        assert index < self.activation.shape[0]
        activation = self.activation[index]  # [batch_size,512]        
        return activation
    def __len__(self):
        return self.activation.shape[0]


class ConceptDataset(Dataset):
    """建立所有概念在某一个特征层上输出的数据集"""
    def __init__(self,datasets:List[SingleConceptDataset],labels:List[int]) -> None:
        """
        Creates the ConceptDataset given a list of K Datasets, and a length K
        list of integer labels representing K different concepts.
        The assumption is that the k-th Dataset of datasets is associated with
        the k-th element of labels.
        The ConceptDataset is the concatenation of the K Datasets in datasets.
        However, __get_item__ not only returns a batch of activation vectors,
        but also a batch of labels indicating which concept that batch of
        activation vectors is associated with.
        Args:
            datasets (list[Dataset]): The k-th element of datasets is a Dataset
                    representing activation vectors associated with the k-th
                    concept
            labels (list[Int]): The k-th element of labels is the integer label
                    associated with the k-th concept
        """
        assert len(datasets) == len(labels),"number of datasets does not match the number of concepts"
        from itertools import accumulate

        offsets = [0] + list(accumulate(map(len, datasets), (lambda x, y: x + y)))
        self.length = offsets[-1]
        self.datasets = datasets
        self.labels = labels
        self.lowers = offsets[:-1]
        self.uppers = offsets[1:]

    def _i_to_k(self, i):

        left, right = 0, len(self.uppers)
        while left < right:
            mid = (left + right) // 2
            if self.lowers[mid] <= i and i < self.uppers[mid]:
                return mid
            if i >= self.uppers[mid]:
                left = mid
            else:
                right = mid

    def __getitem__(self, i):
        """
        Returns a batch of activation vectors, as well as a batch of labels
        indicating which concept the batch of activation vectors is associated
        with.

        args:
            i (int): which (activation vector, label) batch in the dataset to
                    return
        returns:
            inputs (Tensor): i-th batch in Dataset (representing activation
                    vectors)
            labels (Tensor): labels of i-th batch in Dataset
        """
        assert i < self.length
        k = self._i_to_k(i)
        inputs = self.datasets[k][i - self.lowers[k]].unsqueeze(0)
        assert len(inputs.shape) == 2

        labels = torch.tensor([self.labels[k]] * inputs.size(0), device=inputs.device)
        return inputs, labels

    def __len__(self):
        """
        returns the total number of batches in the labelled_dataset
        """
        return self.length
    

# import numpy as np
# concept_dict = np.load("/data/aaron/ACE/ACE_torch/20221104_1423_47_zebra/concepts/concept_info.npy",allow_pickle=True).item()
# dataset_list = []
# label_list = []
# for id,concept_name in enumerate(concept_dict['inception4c']['concepts']):
#     activation = concept_dict['inception4c'][concept_name]['activations']
#     single_dataset = SingleConceptDataset(activation)
#     dataset_list.append(single_dataset)
#     label_list.append(id)

# concept_dataset = ConceptDataset(dataset_list,label_list)
# concept_dataloader = DataLoader(concept_dataset,batch_size=4)
# for input,label in concept_dataloader:
#     print(input.shape)
#     print(label)
