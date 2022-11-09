from collections import defaultdict
from typing import List,Dict,cast
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from captum.concept import Classifier

from .cav import CAV
from .concept import Concept,concepts_to_str
from .concept_dataset import SingleConceptDataset,ConceptDataset

class TrainConcept:
    def __init__(self,activation_path:str,random_path:str):
        self.concept_dict = np.load(activation_path,allow_pickle=True).item()
        self.random_dict = np.load(random_path,allow_pickle=True).item()

        self.layers = []
        for concept_key in list(self.concept_dict.keys()):
            if concept_key != "discovery_images":
                self.layers.append(concept_key)

    
    def batch_collate(self,batch):
        inputs = zip(*batch)
        return torch.cat(inputs)
    
    def define_concepts(self):
        concept_set = {}
        dataset_set = {}


        for layer in self.layers:
            concept_list = []
            dataset_list = []

            for id,concept_name in enumerate(self.concept_dict[layer]['concepts']):
                activation = self.concept_dict[layer][concept_name]['activations']
                single_dataset = SingleConceptDataset(activation)
                single_concept_loader = DataLoader(single_dataset,collate_fn=self.batch_collate)
                concept = Concept(id,concept_name,single_concept_loader)
                dataset_list.append(single_dataset)
                concept_list.append(concept)

            concept_set[layer] = concept_list  # [Concept_1,Concept_2,Concept_3,...,Concept_n]
            dataset_set[layer] = dataset_list  # [Dataset_1,Dataset_2,Dataset_3,...,Dataset_n]

    
        return concept_set,dataset_set
    
    def get_gradients(self):
        gradient_set = {}
        for layer in self.layers:
            gradient_list = []
            for id,concept_name in enumerate(self.concept_dict[layer]['concepts']):
                gradient = self.concept_dict[layer][concept_name]['gradients']
                gradient_list.append(gradient.reshape(gradient.shape[0],-1))
            
            gradient_set[layer] = gradient_list
        
        return gradient_set

    def get_random_sets(self):

        random_set = {}
        random_dataset_set = {}
        random_name = list(self.random_dict.keys())
        layers = list(self.random_dict[random_name[0]].keys())
        new_random_dict = {}
        for layer in layers:
            new_random_dict[layer] = defaultdict()
        
        # step1: 将字典中的key与value的值进行对调
        for random in random_name:
            for layer in layers:
                value = self.random_dict[random][layer]
                new_random_dict[layer].update({random:value})
        # step2: 遍历字典
        self.random_dict = new_random_dict

        for layer in layers:
            random_concept_list = []
            random_dataset_list = []
            for i,random in enumerate(random_name):
                random_data = self.random_dict[layer][random]
                single_random_dataset = SingleConceptDataset(random_data)
                single_random_loader = DataLoader(single_random_dataset,collate_fn=self.batch_collate)
                single_random_concept = Concept(i,random,single_random_loader)
                
                random_concept_list.append(single_random_concept)
                random_dataset_list.append(single_random_dataset)

            random_set[layer] = random_concept_list
            random_dataset_set[layer] = random_dataset_list 
        return random_set,random_dataset_set
    
    def get_concept_keys(self):

        concept_set,dataset_set = self.define_concepts()
        random_set,random_dataset_set = self.get_random_sets()

        concepts_key_list = []
        for i in range(len(concept_set[self.layers[0]])):
            concept_list = [concept_set[self.layers[0]][i],random_set[self.layers[0]][i]]
            concepts_key = concepts_to_str(concept_list) # define concept key
            concepts_key_list.append(concepts_key)
        
        return concepts_key_list


    def train_cav(
        self,
        model_id,
        classifier: Classifier,
        save_path: str,
        classifier_kwargs: Dict,
        save: str = True
    ) -> Dict[str, Dict[str, CAV]]:
        r"""
        A helper function for parallel CAV computations that can be called
        from a python process.

        Please see the TCAV class documentation for further information.

        Args:
            model_id (str): A unique identifier for the PyTorch model for which
                    we would like to load the layer activations and train a
                    model in order to compute CAVs.
            concepts (list[Concept]): A list of Concept objects that are used
                    to train a classifier and learn decision boundaries between
                    those concepts for each layer defined in the `layers`
                    argument.
            layers (str, list[str]): A list of layer names or a single layer
                    name that is used to compute the activations of all concept
                    examples per concept and train a classifier using those
                    activations.
            classifier (Classifier): A custom classifier class, such as the
                    Sklearn "linear_model" that allows us to train a model
                    using the activation vectors extracted for a layer per concept.
                    It also allows us to access trained weights of the classifier
                    and the list of prediction classes.
            save_path (str): The path for storing Concept Activation
                    Vectors (CAVs) and Activation Vectors (AVs).
            classifier_kwargs (dict): Additional named arguments that are passed to
                    concept classifier's `train_and_eval` method.

        Returns:
            cavs (dict): A dictionary of CAV objects indexed by concept ids and
                    layer names. It gives access to the weights of each concept
                    in a given layer and model statistics such as accuracies
                    that resulted in trained concept weights.
        """
        

        concept_set,dataset_set = self.define_concepts()
        random_set,random_dataset_set = self.get_random_sets()

        cavs: Dict[str, Dict[str, CAV]] = defaultdict()

        for i in range(len(concept_set[self.layers[0]])):
            concept_list = [concept_set[self.layers[0]][i],random_set[self.layers[0]][i]]
            concepts_key = concepts_to_str(concept_list) # define concept key
            cavs[concepts_key] = defaultdict()


        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for layer in self.layers:

            for i in range(len(concept_set[layer])):

                concept_list = [concept_set[layer][i],random_set[layer][i]]
                concepts_key = concepts_to_str(concept_list) # define concept key

                # Create data loader to initialize the trainer.
                labels = [0,1]
                dataset_lists = [dataset_set[layer][i],random_dataset_set[layer][i]]
                labelled_dataset = ConceptDataset(cast(List[SingleConceptDataset], dataset_lists), labels)
                def batch_collate(batch):
                    inputs, labels = zip(*batch)
                    return torch.cat(inputs), torch.cat(labels)

                dataloader = DataLoader(labelled_dataset, collate_fn=batch_collate)

                classifier_stats_dict = classifier.train_and_eval(dataloader=dataloader, **classifier_kwargs)
                classifier_stats_dict = (
                    {} if classifier_stats_dict is None else classifier_stats_dict
                )

                weights = classifier.weights()  # 获取权重，也就是每个类别CAV的方向
                assert (
                    weights is not None and len(weights) > 0
                ), "Model weights connot be None or empty"

                classes = classifier.classes()  # 获取对应的类别
                assert (
                    classes is not None and len(classes) > 0
                ), "Classes cannot be None or empty"

                classes = (
                    cast(torch.Tensor, classes).detach().numpy()
                    if isinstance(classes, torch.Tensor)
                    else classes
                )
                cavs[concepts_key][layer] = CAV(    # 将其保存在字典中
                    concept_list,
                    layer,
                    {"weights": weights, "classes": classes, **classifier_stats_dict},
                    save_path,
                    model_id,
                )

        # Saving cavs on the disk
        if save:
            # cavs[concepts_key][layer].save()
            save_dir = os.path.join(save_path,model_id)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_path = os.path.join(save_path,model_id,"train_concept.pkl")
            torch.save(cavs,save_path)

        return cavs