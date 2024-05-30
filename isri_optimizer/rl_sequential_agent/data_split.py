from _dir_init import *
import pickle
from typing import Dict
from sklearn.model_selection import train_test_split
import random

class train_test_split():
    def __init__(self, min_length=20, max_length=100, path="IsriDataDict.pkl", N_TRAINING_INSTANCES=500, all_data=False):
        self.min_length = min_length
        self.max_length = max_length
        self.GA_SOLUTIONS_PATH = path
        self.isri_dataset = pickle.load(open(self.GA_SOLUTIONS_PATH, 'rb'))
        self.all_jobdata_entries = self.isri_dataset['Jobdata']
        if all_data:
            self.all_indices = list(range(len(self.isri_dataset['Jobdata'])))
        else:
            self.all_indices = list(range(N_TRAINING_INSTANCES))
        self.train_indices, self.test_indices = train_test_split(self.all_indices, test_size=0.2, random_state=42) 

    def adjust_entry_length(self, entry):
        target_length = random.randint(self.min_length, self.max_length)
        current_length = len(entry)
    
        if current_length < target_length:
            additional_entries = []
            for other_entry in self.all_jobdata_entries:
                if len(other_entry) > 0 and other_entry != entry:
                    additional_entries.extend(other_entry.items())
                    if len(entry) + len(additional_entries) >= target_length:
                        break
            additional_entries = additional_entries[:target_length - current_length]
            entry.update(additional_entries)

        elif current_length > target_length:
            keys_to_keep = random.sample(list(entry.keys()), target_length)
            entry = {key: entry[key] for key in keys_to_keep}

        return entry

    def get_data(self):
        isri_dataset_train = {}
        isri_dataset_test = {}

        isri_dataset_train.data['Jobdata'] = [self.isri_dataset.data['Jobdata'][i] for i in self.train_indices]
        isri_dataset_train.data['Files'] = [self.isri_dataset.data['Files'][i] for i in self.train_indices]
        isri_dataset_train.data['GAChromosome'] = [self.isri_dataset.data['GAChromosome'][i] for i in self.train_indices]
        isri_dataset_train.data['GAFitness'] = [self.isri_dataset.data['GAFitness'][i] for i in self.train_indices]

        isri_dataset_test.data['Jobdata'] = [self.isri_dataset.data['Jobdata'][i] for i in self.test_indices]
        isri_dataset_test.data['Files'] = [self.isri_dataset.data['Files'][i] for i in self.test_indices]
        isri_dataset_test.data['GAChromosome'] = [self.isri_dataset.data['GAChromosome'][i] for i in self.test_indices]
        isri_dataset_test.data['GAFitness'] = [self.isri_dataset.data['GAFitness'][i] for i in self.test_indices]
        return isri_dataset_train, isri_dataset_test

    def get_mixed_data(self):
        isri_dataset_train = {'Jobdata': [self.adjust_entry_length(self.isri_dataset.data['Jobdata'][i]) for i in self.train_indices]}
        isri_dataset_test = {'Jobdata': [self.adjust_entry_length(self.isri_dataset.data['Jobdata'][i]) for i in self.test_indices]}
        return isri_dataset_train, isri_dataset_test
