from collections import defaultdict
from typing import Union,List,Any,Set,Dict,cast,Tuple
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.concept import Classifier
from captum.log import log_usage
import os
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from captum.attr import LayerActivation, LayerAttribution, LayerGradientXActivation
import numpy as np
import multiprocessing

from .train_concept import TrainConcept
from .concept import ConceptInterpreter,Concept,concepts_to_str
from .cav import CAV
from .classifier import CustomClassifier
from.concept_dataset import SingleConceptDataset,ConceptDataset


class TCAVCompute(ConceptInterpreter):
    r"""
    This class implements ConceptInterpreter abstract class using an
    approach called Testing with Concept Activation Vectors (TCAVs),
    as described in the paper:
    https://arxiv.org/pdf/1711.11279.pdf

    TCAV scores for a given layer, a list of concepts and input example
    are computed using the dot product between prediction's layer
    sensitivities for given input examples and Concept Activation Vectors
    (CAVs) in that same layer.

    CAVs are defined as vectors that are orthogonal to the classification boundary
    hyperplane that separate given concepts in a given layer from each other.
    For a given layer, CAVs are computed by training a classifier that uses the
    layer activation vectors for a set of concept examples as input examples and
    concept ids as corresponding input labels. Trained weights of
    that classifier represent CAVs.

    CAVs are represented as a learned weight matrix with the dimensionality
    C X F, where:
    F represents the number of input features in the classifier.
    C is the number of concepts used for the classification. Concept
    ids are used as labels for concept examples during the training.

    We can use any layer attribution algorithm to compute layer sensitivities
    of a model prediction.
    For example, the gradients of an output prediction w.r.t. the outputs of
    the layer.
    The CAVs and the Sensitivities (SENS) are used to compute the TCAV score:

    0. TCAV = CAV • SENS, a dot product between those two vectors

    The final TCAV score can be computed by aggregating the TCAV scores
    for each input concept based on the sign or magnitude of the tcav scores.

    1. sign_count_score = | TCAV > 0 | / | TCAV |
    2. magnitude_score = SUM(ABS(TCAV * (TCAV > 0))) / SUM(ABS(TCAV))
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Union[str, List[str]],
        activation_path : str,
        random_path: str,
        model_id: str = "default_model_id",
        classifier: Classifier = None,
        layer_attr_method: LayerAttribution = None,
        attribute_to_layer_input=False,
        save_path: str = "./cav/",
        **classifier_kwargs: Any,
    ) -> None:
        r"""
        Args:
            model (Module): An instance of pytorch model that is used to compute
                    layer activations and attributions.
            layers (str, list[str]): A list of layer name(s) that are
                    used for computing concept activations (cavs) and layer
                    attributions.
            model_id (str, optional): A unique identifier for the PyTorch `model`
                    passed as first argument to the constructor of TCAV class. It
                    is used to store and load activations for given input `model`
                    and associated `layers`.
            classifier (Classifier, optional): A custom classifier class, such as the
                    Sklearn "linear_model" that allows us to train a model
                    using the activation vectors extracted for a layer per concept.
                    It also allows us to access trained weights of the model
                    and the list of prediction classes.
            layer_attr_method (LayerAttribution, optional): An instance of a layer
                    attribution algorithm that helps us to compute model prediction
                    sensitivity scores.

                    Default: None
                    If `layer_attr_method` is None, we default it to gradients
                    for the layers using `LayerGradientXActivation` layer
                    attribution algorithm.
            save_path (str, optional): The path for storing CAVs and
                    Activation Vectors (AVs).
            classifier_kwargs (any, optional): Additional arguments such as
                    `test_split_ratio` that are passed to concept `classifier`.

        Examples::
            >>>
            >>> # TCAV use example:
            >>>
            >>> # Define the concepts
            >>> stripes = Concept(0, "stripes", striped_data_iter)
            >>> random = Concept(1, "random", random_data_iter)
            >>>
            >>>
            >>> mytcav = TCAV(model=imagenet,
            >>>     layers=['inception4c', 'inception4d'])
            >>>
            >>> scores = mytcav.interpret(inputs, [[stripes, random]], target = 0)
            >>>
            For more thorough examples, please check out TCAV tutorial and test cases.
        """
        ConceptInterpreter.__init__(self, model)
        self.layers = [layers] if isinstance(layers, str) else layers  # 特征层
        self.model_id = model_id           # model的id，默认为""default_model_id"
        self.concepts: Set[Concept] = set()  # 获取概念的集合
        self.classifier = classifier    # 获取分类器
        self.classifier_kwargs = classifier_kwargs  # 分类器的一些参数
        self.cavs: Dict[str, Dict[str, CAV]] = defaultdict(lambda: defaultdict())  # 建立一个空的储存CAV的字典
        if self.classifier is None:
            self.classifier = CustomClassifier()  # 如果没有定义classifer就用默认的

        else:
            self.layer_attr_method = layer_attr_method

        assert model_id, (
            "`model_id` cannot be None or empty. Consider giving `model_id` "
            "a meaningful name or leave it unspecified. If model_id is unspecified we "
            "will use `default_model_id` as its default value."
        )

        self.attribute_to_layer_input = attribute_to_layer_input
        self.save_path = save_path  # 获取保存的路径

        self.train_concept = TrainConcept(activation_path,random_path)

        # Creates CAV save directory if it doesn't exist. It is created once in the
        # constructor before generating the CAVs.
        # It is assumed that `model_id` can be used as a valid directory name
        # otherwise `create_cav_dir_if_missing` will raise an error
        CAV.create_cav_dir_if_missing(self.save_path, model_id)  # 建立概念保存的路径地址


    def compute_cavs(
        self,
        force_train: bool = False,
        save: bool = True,
        save_path: str = None,
        weight_path: str = None
    ):
        r"""
        This method computes CAVs for given `experiments_sets` and layers
        specified in `self.layers` instance variable. Internally, it
        trains a classifier and creates an instance of CAV class using the
        weights of the trained classifier for each experimental set.

        It also allows to compute the CAVs in parallel using python's
        multiprocessing API and the number of processes specified in
        the argument.

        Args:
            experimental_sets (list[list[Concept]]): A list of lists of concept
                    instances for which the cavs will be computed.
            force_train (bool, optional): A flag that indicates whether to
                    train the CAVs regardless of whether they are saved or not.
                    Default: False
            processes (int, optional): The number of processes to be created
                    when running in multi-processing mode. If processes > 0 then
                    CAV computation will be performed in parallel using
                    multi-processing, otherwise it will be performed sequentially
                    in a single process.
                    Default: None
        Returns:
            cavs (dict) : A mapping of concept ids and layers to CAV objects.
                    If CAVs for the concept_ids-layer pairs are present in the
                    data storage they will be loaded into the memory, otherwise
                    they will be computed using a training process and stored
                    in the data storage that can be configured using `save_path`
                    input argument.
        """

        if not force_train:  # 如果不是force_train，说明已经训练好了，直接导入相关权重文件即可
            assert weight_path is not None
            # concepts_key_list = self.train_concept.get_concept_keys()

            # for concepts_key in concepts_key_list:
            #     for layer in self.layers:
            #         file_name = concepts_key + "-" + layer + ".pkl"
            #         concept_checkpoints = torch.load(os.path.join(weight_path,file_name))
            #         self.cavs[concepts_key].update(concept_checkpoints)
            file_name = 'train_concept.pkl'
            cavs = torch.load(os.path.join(weight_path,self.model_id,file_name))
            
        else:
            if save_path is None and weight_path is not None:
                save_path = weight_path
            elif save_path is None and weight_path is None:
                raise "save path and weight path are all None"
            else:
                cavs = self.train_concept.train_cav(self.model_id,self.classifier,
                    save_path,self.classifier_kwargs,save)
            

        self.cavs = cavs

        return self.cavs
    

    @log_usage()
    def interpret(
        self,
        force_train: bool = True,
        save: bool = True,  # 保存训练方向导数的文件
        save_path: str = None, # 保存训练方向导数的文件路径地址
        weight_path: str = None, # 如果force_train为false，就导入已经训练好的权重
        sort:bool = True,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, Dict[str, Tensor]]]:
        r"""
        This method computes magnitude and sign-based TCAV scores for each
        experimental sets in `experimental_sets` list.
        TCAV scores are computed using a dot product between layer attribution
        scores for specific predictions and CAV vectors.

        Args:
            inputs (tensor or tuple of tensors): Inputs for which predictions
                    are performed and attributions are computed.
                    If model takes a single tensor as
                    input, a single input tensor should be provided.
                    If model takes multiple tensors as
                    input, a tuple of the input tensors should be provided.
                    It is assumed that for all given input tensors,
                    dimension 0 corresponds to the number of examples
                    (aka batch size), and if multiple input tensors are
                    provided, the examples must be aligned appropriately.
            experimental_sets (list[list[Concept]]): A list of list of Concept
                    instances.
            target (int, tuple, tensor or list, optional):  Output indices for
                    which attributions are computed (for classification cases,
                    this is usually the target class).
                    If the network returns a scalar value per example,
                    no target index is necessary.
                    For general 2D outputs, targets can be either:

                    - a single integer or a tensor containing a single
                        integer, which is applied to all input examples
                    - a list of integers or a 1D tensor, with length matching
                        the number of examples in inputs (dim 0). Each integer
                        is applied as the target for the corresponding example.

                    For outputs with > 2 dimensions, targets can be either:

                    - A single tuple, which contains #output_dims - 1
                        elements. This target index is applied to all examples.
                    - A list of tuples with length equal to the number of
                        examples in inputs (dim 0), and each tuple containing
                        #output_dims - 1 elements. Each tuple is applied as the
                        target for the corresponding example.

            additional_forward_args (Any, optional): Extra arguments that are passed to
                     model when computing the attributions for `inputs`
                     w.r.t. layer output.
                     Default: None
            processes (int, optional): The number of processes to be created. if
                    processes is larger than one then CAV computations will be
                    performed in parallel using the number of processes equal to
                    `processes`. Otherwise, CAV computations will be performed
                    sequential.
                    Default:None
            **kwargs (Any, optional): A list of arguments that are passed to layer
                    attribution algorithm's attribute method. This could be for
                    example `n_steps` in case of integrated gradients.
                    Default: None
        Returns:
            results (dict): A dictionary of sign and magnitude -based tcav scores
                    for each concept set per layer.
                    The order of TCAV scores in the resulting tensor for each
                    experimental set follows the order in which concepts
                    are passed in `experimental_sets` input argument.

        results example::
            >>> #
            >>> # scores =
            >>> # {'0-1':
            >>> #     {'inception4c':
            >>> #         {'sign_count': tensor([0.5800, 0.4200]),
            >>> #          'magnitude': tensor([0.6613, 0.3387])},
            >>> #      'inception4d':
            >>> #         {'sign_count': tensor([0.6200, 0.3800]),
            >>> #           'magnitude': tensor([0.7707, 0.2293])}}),
            >>> #  '0-2':
            >>> #     {'inception4c':
            >>> #         {'sign_count': tensor([0.6200, 0.3800]),
            >>> #          'magnitude': tensor([0.6806, 0.3194])},
            >>> #      'inception4d':
            >>> #         {'sign_count': tensor([0.6400, 0.3600]),
            >>> #          'magnitude': tensor([0.6563, 0.3437])}})})
            >>> #

        """
        assert "attribute_to_layer_input" not in kwargs, (
            "Please, set `attribute_to_layer_input` flag as a constructor "
            "argument to TCAV class. In that case it will be applied "
            "consistently to both layer activation and layer attribution methods."
        )
        self.compute_cavs(force_train,save,save_path,weight_path)

        experimental_sets = []
        for concepts_keys in self.cavs.keys():
            for layer in self.layers:
                concept_list = self.cavs[concepts_keys][layer].concepts
                experimental_sets.append(concept_list)

        scores: Dict[str, Dict[str, Dict[str, Tensor]]] = defaultdict(
            lambda: defaultdict()
        )

        attribs_dict = self.train_concept.get_gradients()
        for layer in self.layers:
            
            attribs_list = [torch.tensor(attribs_tensor) for attribs_tensor in attribs_dict[layer]]
            # n_inputs x n_features

            # n_experiments x n_concepts x n_features
            cavs = []
            accs = []
            for concepts in experimental_sets:
                concepts_key = concepts_to_str(concepts)
                cavs_stats = cast(Dict[str, Any], self.cavs[concepts_key][layer].stats)
                cavs.append(cavs_stats["weights"].float().detach().tolist())
                accs.append(cavs_stats["accs"].item())  # 保存训练的精度

            i = 0
            while i < len(experimental_sets):
                cav_subset = cavs[i]
                acc = accs[i]
                # n_experiments x n_concepts x n_features
                cav_subset = torch.tensor(cav_subset)
                attribs = attribs_list[i]
                cav_subset = cav_subset.to(attribs.device)



                experimental_subset = experimental_sets[i]
                self._tcav_sub_computation(
                    scores,
                    layer,
                    attribs,
                    cav_subset,
                    experimental_subset,
                    acc=acc
                )
                i += 1
        if sort:
            scores = self._sort_concepts(scores)

        return scores

    def _tcav_sub_computation(
        self,
        scores: Dict[str, Dict[str, Dict[str, Tensor]]],
        layer: str,
        attribs: Tensor,
        cavs: Tensor,
        experimental_set: List[Concept],  # [Concept_1,Concept_2]
        acc: float,
        complicated: bool=False # 是否输出两种类型的TCAV值，默认为false
    ) -> None:
        # n_inputs x n_concepts
        tcav_score = torch.matmul(attribs.float(), cavs.transpose(1,0))


        assert attribs.shape[0] == tcav_score.shape[0], (     # [40,2]
            "attrib and tcav_score should have the same 1st "
            "dimensions respectively (n_inputs)."
        )
        # n_experiments x n_concepts
        sign_count_score = torch.mean((tcav_score > 0.0).float(), dim=0)

        magnitude_score = torch.mean(tcav_score, dim=0)

        concepts_key = concepts_to_str(experimental_set)

        concept_name = experimental_set[0].name

        if complicated:
            scores[layer][concept_name] = {
                "sign_count": sign_count_score,
                "magnitude": magnitude_score,
                "acc":acc,

            }
        else:
            scores[layer][concept_name] = {
                'tcav': sign_count_score[0],
                "acc": acc
            }

    def _sort_concepts(self,scores):
        for layer in self.layers:
            tcavs = []
            for concept in scores[layer].keys():
                single_tcav = scores[layer][concept]['tcav']
                tcavs.append(single_tcav.item())
            concepts = []
            for index in np.argsort(tcavs)[::-1]:
                concepts.append(list(scores[layer].keys())[index])
             
            scores[layer]['concepts'] = concepts

        return scores


