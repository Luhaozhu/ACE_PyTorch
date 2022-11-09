#!/usr/bin/env python3

from typing import Callable, Union, List  # 类型注解函数

import torch
from torch.nn import Module


class Concept:  # 用来保存概念的数据集，每一个概念定义一个数据集

    r"""
    Concepts are human-friendly abstract representations that can be
    numerically encoded into torch tensors. They can be illustrated as
    images, text or any other form of representation. In case of images,
    for example, "stripes" concept can be represented through a number
    of example images resembling "stripes" in various different
    contexts. In case of Natural Language Processing, the concept of
    "happy", for instance, can be illustrated through a number of
    adjectives and words that convey happiness.
    """

    def __init__(
        self, id: int, name: str, data_iter: Union[None, torch.utils.data.DataLoader]
    ) -> None:

        r"""
        Args:
            id (int):   The unique identifier of the concept.概念的索引值
            name (str): A unique name of the concept.  概念的名称
            data_iter (DataLoader): A pytorch DataLoader object that combines a dataset
                        and a sampler, and provides an iterable over a given
                        dataset. Only the input batches are provided by `data_iter`.
                        Concept ids can be used as labels if necessary.
                        For more information, please check:
                        https://pytorch.org/docs/stable/data.html  概念定义的DataLoader

        Example::
            >>> # Creates a Concept object named "striped", with a data_iter
            >>> # object to iterate over all files in "./concepts/striped"
            >>> concept_name = "striped"
            >>> concept_path = os.path.join("./concepts", concept_name) + "/"
            >>> concept_iter = dataset_to_dataloader(
            >>> get_tensor_from_filename, concepts_path=concept_path)
            >>> concept_object = Concept(
                    id=0, name=concept_name, data_iter=concept_iter)
        """

        self.id = id
        self.name = name
        self.data_iter = data_iter

    @property
    def identifier(self) -> str:
        return "%s-%s" % (self.name, self.id)

    def __repr__(self) -> str: # 都是实例可视化显示  print(Concept)会显示这个
        return "Concept(%r, %r)" % (self.id, self.name)


class ConceptInterpreter:
    r"""
    An abstract class that exposes an abstract interpret method
    that has to be implemented by a specific algorithm for
    concept-based model interpretability.
    """

    def __init__(self, model: Module) -> None:
        r"""
        Args:
            model (torch.nn.Module): An instance of pytorch model.
        """
        self.model = model

    interpret: Callable
    r"""
    An abstract interpret method that performs concept-based model interpretability
    and returns the interpretation results in form of tensors, dictionaries or other
    data structures.

    Args:

        inputs (tensor or tuple of tensors):  Inputs for which concept-based
                    interpretation scores are computed. It can be provided as
                    a single tensor or a tuple of multiple tensors. If multiple
                    input tensors are provided, the batch size (the first
                    dimension of the tensors) must be aligned across all tensors.
    """

def concepts_to_str(concepts: List[Concept]) -> str:
    r"""
    Returns a string of hyphen("-") concatenated concept names.
    Example output: "striped-random_0-random_1"

    Args:
        concepts (list[Concept]): a List of concept names to be
                concatenated and used as a concepts key. These concept
                names are respective to the Concept objects used for
                the classifier train.
    Returns:
        names_str (str): A string of hyphen("-") concatenated
                concept names. Ex.: "striped-random_0-random_1"
    """

    return "-".join([str(c.id) for c in concepts])