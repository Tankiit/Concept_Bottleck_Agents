import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import Dict
import openai
from openai import OpenAI
import time
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.cluster import KMeans
import torch.nn.functional as F

openai.api_key="sk-q9AsaKU3isM9Oh6tZ61vawXJntj7ddYgPnooD9Ns6CT3BlbkFJVbONDlzhfRO9rF4mM9HiXrYqrdURsM25-1aE0fL9gA"

class SelectionAgent:
    def __init__(self, strategy: str,organization: str, project: str):
        self.strategy = strategy
        self.client = OpenAI(api_key='sk-q9AsaKU3isM9Oh6tZ61vawXJntj7ddYgPnooD9Ns6CT3BlbkFJVbONDlzhfRO9rF4mM9HiXrYqrdURsM25-1aE0fL9gA',
            organization=organization,
            project=project
        )

    def propose_selection(self, model_performance: float, curriculum_stage: float, class_distribution: Dict[str, int]) -> str:
        prompt = f"""
        As the {self.strategy} agent for CIFAR-10 image classification:
        Current model accuracy: {model_performance:.2f}
        Current curriculum stage: {curriculum_stage:.2f}
        Class distribution: {class_distribution}
        Propose how to select the next batch of samples and why. Consider:
        1. The strengths of your strategy ({self.strategy})
        2. The current model performance and curriculum stage
        3. The current class distribution
        Respond in 50 words or less.
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are the {self.strategy} selection agent for CIFAR-10 curriculum learning."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=75,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

class ModeratorAgent:
    def __init__(self,organization: str, project: str):
        self.client = OpenAI(api_key='sk-q9AsaKU3isM9Oh6tZ61vawXJntj7ddYgPnooD9Ns6CT3BlbkFJVbONDlzhfRO9rF4mM9HiXrYqrdURsM25-1aE0fL9gA',
            organization=organization,
            project=project
        )

    def moderate_discussion(self, agent_proposals: Dict[str, str], model_performance: float, curriculum_stage: float, current_weights: Dict[str, float]) -> Dict[str, float]:
        proposals = "\n".join([f"{agent}: {proposal}" for agent, proposal in agent_proposals.items()])
        current_weights_str = ", ".join([f"{k}: {v:.2f}" for k, v in current_weights.items()])
        prompt = f"""
        As the moderator, review the proposals from different selection agents:
        {proposals}
        Current model accuracy: {model_performance:.2f}
        Current curriculum stage: {curriculum_stage:.2f}
        Current strategy weights: {current_weights_str}
        Suggest adjustments to the current weights based on the proposals and current state. The weights should sum to 1.
        Respond with a Python dictionary: {{"S_U": float, "S_C": float, "S_B": float, "S_D": float}}
        Provide a brief explanation for your decision.
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "system", "content": "You are the moderator for CIFAR-10 curriculum learning strategy selection."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
        )
        result = response.choices[0].message.content.strip()
    
        # Try to extract the dictionary from the response
        try:
            dict_start = result.index('{')
            dict_end = result.index('}', dict_start) + 1
            dict_str = result[dict_start:dict_end]
            # Attempt to fix incomplete dictionary
            if dict_str[-1] != '}':
              dict_str += '}'
            suggested_weights = eval(dict_str)
            explanation = result[dict_end:].strip()
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing moderator's response: {e}")
            print("Moderator's response:", result)
            suggested_weights = current_weights  # Fall back to current weights
            explanation = "Error in parsing moderator's response. Using current weights."
    
        # Ensure all required keys are present and values are valid
        for key in ["S_U", "S_C", "S_B", "S_D"]:
            if key not in suggested_weights or not isinstance(suggested_weights[key], (int, float)):
                suggested_weights[key] = current_weights.get(key, 0.25)
    
        # Normalize weights to ensure they sum to 1
        total = sum(suggested_weights.values())
        suggested_weights = {k: v/total for k, v in suggested_weights.items()}
    
        return suggested_weights, explanation
        
class MetaAgent:
      def __init__(self, strategies, learning_rate=0.1):
        self.strategies = strategies
        self.weights = {s: 1.0 / len(strategies) for s in strategies}
        self.learning_rate = learning_rate
        self.performance_history = {s: [] for s in strategies}

      def update_weights_with_moderator(self, rewards: Dict[str, float], suggested_weights: Dict[str, float]):
          for strategy in self.strategies:
              current_weight = self.weights[strategy]
              reward = rewards[strategy]
              suggested_weight = suggested_weights[strategy]
            
              # Combine current weight, reward-based update, and moderator suggestion
              new_weight = current_weight * (1 + self.learning_rate * reward)
              new_weight = 0.3 * new_weight + 0.7 * suggested_weight  # More emphasis on moderator
            
              self.weights[strategy] = max(0.1, new_weight)  # Ensure a minimum weight

          # Normalize weights
          total_weight = sum(self.weights.values())
          for strategy in self.strategies:
              self.weights[strategy] /= total_weight

          print(f"Updated weights: {self.weights}")

      def get_weights(self):
          return self.weights
class CurriculumCIFAR10Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_criterion()
        self.setup_strategies()
        self.setup_agents()
        self.curriculum_stage = 0.0
        self.last_accuracy = 0.0
        self.metrics = []
        self.start_time = time.time()
        self.previous_indices = []
        self.kmeans=None
        self.meta_agent = MetaAgent(list(self.strategies.keys()))
        self.setup_feature_extractor()

        # self.args = args
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.setup_data()
        # self.setup_model()
        # self.setup_optimizer()
        # self.setup_criterion()
        # self.setup_strategies()
        # self.setup_agents()
        # self.curriculum_stage = 0.0
        # self.last_accuracy = 0.0
        # self.metrics = []
        # self.start_time = time.time()

    def setup_feature_extractor(self):
        # Create a new model for feature extraction
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor.eval()  # Set to evaluation mode

    def extract_features(self, indices):
        self.feature_extractor.eval()
        features = []
        subset = Subset(self.full_dataset, indices)
        dataloader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=self.args.workers)
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                feature = self.feature_extractor(inputs)
                features.append(feature.squeeze().cpu().numpy())
        
        return np.vstack(features)

    def setup_data(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        self.test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        
        self.current_indices = self.smart_initial_sampling(self.args.initial_sample_size)
        self.update_train_loader()
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.workers)

    def setup_model(self):
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        self.model = self.model.to(self.device)
    def get_weights(self):
        return self.meta_agent.get_weights()


    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)

    def setup_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def setup_strategies(self):
        self.strategies = {
        "S_U": self.get_uncertain_samples,
        "S_C": self.get_class_balanced_samples,
        "S_B": self.get_difficult_samples,  # Changed from get_diverse_samples
        "S_D": self.get_diverse_samples
    }
        
    def calculate_strategy_rewards(self):
         rewards = {}
         for strategy in self.strategies:
            # 1. Performance improvement
            current_accuracy = self.last_accuracy
            previous_accuracy = self.metrics[-2]['test_acc'] if len(self.metrics) > 1 else 0
            accuracy_improvement = max(0, current_accuracy - previous_accuracy)

            # 2. Sample diversity
            new_samples = list(set(self.current_indices) - set(self.previous_indices))
            diversity_score = self.calculate_sample_diversity(new_samples)

            # 3. Class balance improvement
            current_balance = self.calculate_class_balance(self.current_indices)
            previous_balance = self.calculate_class_balance(self.previous_indices)
            balance_improvement = max(0, current_balance - previous_balance)

            # 4. Difficulty progression
            difficulty_scores = self.assess_sample_difficulty(new_samples)
            difficulty_score = np.mean(difficulty_scores)

            # 5. Strategy utilization
            strategy_usage = self.meta_agent.weights[strategy]

            # Combine factors with weights
            reward = (
                0.4 * accuracy_improvement +
                0.2 * diversity_score +
                0.2 * balance_improvement +
                0.1 * difficulty_score +
                0.1 * strategy_usage
            )

            rewards[strategy] = reward

         # Normalize rewards
         min_reward = min(rewards.values())
         max_reward = max(rewards.values())
         if max_reward > min_reward:
            rewards = {s: (r - min_reward) / (max_reward - min_reward) for s, r in rewards.items()}
         else:
             rewards = {s: 1.0 for s in self.strategies}  # If all rewards are equal

         return rewards


    

    def setup_agents(self):
        self.selection_agents = {
            "S_U": SelectionAgent("uncertainty", self.args.organization, self.args.project),
            "S_C": SelectionAgent("class balance", self.args.organization, self.args.project),
            "S_B": SelectionAgent("boundary", self.args.organization, self.args.project),
            "S_D": SelectionAgent("diversity", self.args.organization, self.args.project)
        }
        self.moderator = ModeratorAgent(self.args.organization, self.args.project)
        self.meta_agent = MetaAgent(list(self.strategies.keys()))

    def update_train_loader(self):
        subset = Subset(self.full_dataset, self.current_indices)
        self.train_loader = DataLoader(subset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers)

    def smart_initial_sampling(self, n_samples):
        return np.random.choice(len(self.full_dataset), n_samples, replace=False)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # Check if the batch size is greater than 1
            if inputs.size(0) <= 1:
                print(f"Skipping batch {batch_idx} with size {inputs.size(0)}")
                continue
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % self.args.print_freq == 0:
                print(f'Epoch: {self.current_epoch} [{batch_idx * len(inputs)}/{len(self.train_loader.dataset)} ({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        return total_loss / len(self.train_loader), correct / total

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        accuracy = correct / total
        f1 = f1_score(all_targets, all_preds, average='weighted')
        precision = precision_score(all_targets, all_preds, average='weighted')
        recall = recall_score(all_targets, all_preds, average='weighted')
        conf_matrix = confusion_matrix(all_targets, all_preds)

        return total_loss / len(self.test_loader), accuracy, f1, precision, recall, conf_matrix

    def assess_sample_difficulty(self, samples):
        self.model.eval()
        difficulties = []
        dataloader = DataLoader(Subset(self.full_dataset, samples), batch_size=64, shuffle=False, num_workers=self.args.workers)
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                
                # Method 1: Entropy of predictions
                probabilities = F.softmax(outputs, dim=1)
                entropies = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
                
                # Method 2: 1 - max probability
                max_probs = torch.max(probabilities, dim=1)[0]
                uncertainty = 1 - max_probs
                
                # Method 3: Margin of top two classes
                top2_probs, _ = torch.topk(probabilities, k=2, dim=1)
                margins = top2_probs[:, 0] - top2_probs[:, 1]
                
                # Method 4: Loss value
                losses = F.cross_entropy(outputs, targets, reduction='none')
                
                # Combine methods (you can adjust these weights)
                difficulty = (
                    0.3 * entropies.cpu().numpy() +
                    0.3 * uncertainty.cpu().numpy() +
                    0.2 * (1 - margins.cpu().numpy()) +  # Invert margin so higher value = more difficult
                    0.2 * losses.cpu().numpy()
                )
                
                difficulties.extend(difficulty)

        # Normalize difficulties to [0, 1] range
        difficulties = np.array(difficulties)
        difficulties = (difficulties - difficulties.min()) / (difficulties.max() - difficulties.min() + 1e-10)
        
        return difficulties
    
    def calculate_class_balance(self, indices):
        class_counts = Counter([self.full_dataset[i][1] for i in indices])
        total = sum(class_counts.values())
        class_proportions = [count / total for count in class_counts.values()]
        return 1 - np.std(class_proportions)  # Higher value means better balance
    
    def get_difficult_samples(self, n_samples):
        current_subset = Subset(self.full_dataset, self.current_indices)
        
        # If we don't have enough samples in our current subset, return all indices
        if len(current_subset) <= n_samples:
            return list(range(len(current_subset)))

        difficulties = self.assess_sample_difficulty(range(len(current_subset)))
        
        # Select the n_samples most difficult samples
        difficult_indices = np.argsort(difficulties)[-n_samples:]
        
        return difficult_indices

    def get_uncertain_samples(self, n_samples):
        self.model.eval()
        uncertainties = []
        with torch.no_grad():
            for inputs, _ in DataLoader(self.full_dataset, batch_size=self.args.batch_size):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                uncertainties.extend((-probabilities * torch.log(probabilities)).sum(1).cpu().numpy())
        
        uncertain_indices = np.argsort(uncertainties)[::-1]
        return [idx for idx in uncertain_indices if idx not in self.current_indices][:n_samples]

    def get_class_balanced_samples(self, n_samples):
        class_counts = Counter([self.full_dataset[i][1] for i in self.current_indices])
        samples_per_class = n_samples // 10
        balanced_samples = []
        for class_label in range(10):
            class_samples = [i for i in range(len(self.full_dataset)) if self.full_dataset[i][1] == class_label and i not in self.current_indices]
            balanced_samples.extend(np.random.choice(class_samples, min(samples_per_class, len(class_samples)), replace=False))
        return balanced_samples
        
    


    def get_diverse_samples(self, n_samples):
         n_samples = int(n_samples)
        
         if len(self.current_indices) <= n_samples:
            return self.current_indices

         try:
            features = self.extract_features(self.current_indices)
         except Exception as e:
            print(f"Error in feature extraction: {e}")
            # Fallback to random sampling if feature extraction fails
            return np.random.choice(self.current_indices, size=n_samples, replace=False)

         if features.shape[0] == 0:
            print("No features extracted. Falling back to random sampling.")
            return np.random.choice(self.current_indices, size=n_samples, replace=False)

         scaler = StandardScaler()
         normalized_features = scaler.fit_transform(features)

         n_clusters = min(n_samples, len(self.current_indices))
         self.kmeans = KMeans(n_clusters=n_clusters, n_init=10)
         cluster_labels = self.kmeans.fit_predict(normalized_features)

         diverse_samples = []
         for cluster in range(n_clusters):
             cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
             if cluster_indices:
                cluster_center = self.kmeans.cluster_centers_[cluster]
                distances = [np.linalg.norm(normalized_features[i] - cluster_center) for i in cluster_indices]
                closest_index = cluster_indices[np.argmin(distances)]
                diverse_samples.append(self.current_indices[closest_index])

         if len(diverse_samples) < n_samples:
            remaining = list(set(self.current_indices) - set(diverse_samples))
            diverse_samples.extend(np.random.choice(remaining, n_samples - len(diverse_samples), replace=False))

         return diverse_samples

    def calculate_sample_diversity(self, new_samples):
        if not new_samples:
            return 0

        if self.kmeans is None:
            # If KMeans hasn't been run yet, we can't calculate diversity meaningfully
            return 0

        features = self.extract_features(new_samples)
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)

        # Assign new samples to existing clusters
        cluster_assignments = self.kmeans.predict(normalized_features)

        # Calculate the distribution of new samples across clusters
        cluster_distribution = np.bincount(cluster_assignments, minlength=self.kmeans.n_clusters)
        cluster_distribution = cluster_distribution / len(new_samples)

        # Calculate entropy as a measure of diversity
        entropy = -np.sum(cluster_distribution * np.log(cluster_distribution + 1e-10))

        # Normalize entropy to [0, 1]
        max_entropy = np.log(self.kmeans.n_clusters)
        normalized_entropy = entropy / max_entropy

        return normalized_entropy




    import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import Dict
import openai
from openai import OpenAI
import time
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.cluster import KMeans
import torch.nn.functional as F

openai.api_key="sk-q9AsaKU3isM9Oh6tZ61vawXJntj7ddYgPnooD9Ns6CT3BlbkFJVbONDlzhfRO9rF4mM9HiXrYqrdURsM25-1aE0fL9gA"

class SelectionAgent:
    def __init__(self, strategy: str,organization: str, project: str):
        self.strategy = strategy
        self.client = OpenAI(api_key='sk-q9AsaKU3isM9Oh6tZ61vawXJntj7ddYgPnooD9Ns6CT3BlbkFJVbONDlzhfRO9rF4mM9HiXrYqrdURsM25-1aE0fL9gA',
            organization=organization,
            project=project
        )

    def propose_selection(self, model_performance: float, curriculum_stage: float, class_distribution: Dict[str, int]) -> str:
        prompt = f"""
        As the {self.strategy} agent for CIFAR-10 image classification:
        Current model accuracy: {model_performance:.2f}
        Current curriculum stage: {curriculum_stage:.2f}
        Class distribution: {class_distribution}
        Propose how to select the next batch of samples and why. Consider:
        1. The strengths of your strategy ({self.strategy})
        2. The current model performance and curriculum stage
        3. The current class distribution
        Respond in 50 words or less.
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are the {self.strategy} selection agent for CIFAR-10 curriculum learning."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=75,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

class ModeratorAgent:
    def __init__(self,organization: str, project: str):
        self.client = OpenAI(api_key='sk-q9AsaKU3isM9Oh6tZ61vawXJntj7ddYgPnooD9Ns6CT3BlbkFJVbONDlzhfRO9rF4mM9HiXrYqrdURsM25-1aE0fL9gA',
            organization=organization,
            project=project
        )

    def moderate_discussion(self, agent_proposals: Dict[str, str], model_performance: float, curriculum_stage: float, current_weights: Dict[str, float]) -> Dict[str, float]:
        proposals = "\n".join([f"{agent}: {proposal}" for agent, proposal in agent_proposals.items()])
        current_weights_str = ", ".join([f"{k}: {v:.2f}" for k, v in current_weights.items()])
        prompt = f"""
        As the moderator, review the proposals from different selection agents:
        {proposals}
        Current model accuracy: {model_performance:.2f}
        Current curriculum stage: {curriculum_stage:.2f}
        Current strategy weights: {current_weights_str}
        Suggest adjustments to the current weights based on the proposals and current state. The weights should sum to 1.
        Respond with a Python dictionary: {{"S_U": float, "S_C": float, "S_B": float, "S_D": float}}
        Provide a brief explanation for your decision.
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "system", "content": "You are the moderator for CIFAR-10 curriculum learning strategy selection."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
        )
        result = response.choices[0].message.content.strip()
    
        # Try to extract the dictionary from the response
        try:
            dict_start = result.index('{')
            dict_end = result.index('}', dict_start) + 1
            dict_str = result[dict_start:dict_end]
            # Attempt to fix incomplete dictionary
            if dict_str[-1] != '}':
              dict_str += '}'
            suggested_weights = eval(dict_str)
            explanation = result[dict_end:].strip()
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing moderator's response: {e}")
            print("Moderator's response:", result)
            suggested_weights = current_weights  # Fall back to current weights
            explanation = "Error in parsing moderator's response. Using current weights."
    
        # Ensure all required keys are present and values are valid
        for key in ["S_U", "S_C", "S_B", "S_D"]:
            if key not in suggested_weights or not isinstance(suggested_weights[key], (int, float)):
                suggested_weights[key] = current_weights.get(key, 0.25)
    
        # Normalize weights to ensure they sum to 1
        total = sum(suggested_weights.values())
        suggested_weights = {k: v/total for k, v in suggested_weights.items()}
    
        return suggested_weights, explanation
        
class MetaAgent:
      def __init__(self, strategies, learning_rate=0.1):
        self.strategies = strategies
        self.weights = {s: 1.0 / len(strategies) for s in strategies}
        self.learning_rate = learning_rate
        self.performance_history = {s: [] for s in strategies}

      def update_weights_with_moderator(self, rewards: Dict[str, float], suggested_weights: Dict[str, float]):
          for strategy in self.strategies:
              current_weight = self.weights[strategy]
              reward = rewards[strategy]
              suggested_weight = suggested_weights[strategy]
            
              # Combine current weight, reward-based update, and moderator suggestion
              new_weight = current_weight * (1 + self.learning_rate * reward)
              new_weight = 0.3 * new_weight + 0.7 * suggested_weight  # More emphasis on moderator
            
              self.weights[strategy] = max(0.1, new_weight)  # Ensure a minimum weight

          # Normalize weights
          total_weight = sum(self.weights.values())
          for strategy in self.strategies:
              self.weights[strategy] /= total_weight

          print(f"Updated weights: {self.weights}")

      def get_weights(self):
          return self.weights
class CurriculumCIFAR10Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_criterion()
        self.setup_strategies()
        self.setup_agents()
        self.curriculum_stage = 0.0
        self.last_accuracy = 0.0
        self.metrics = []
        self.start_time = time.time()
        self.previous_indices = []
        self.kmeans=None
        self.meta_agent = MetaAgent(list(self.strategies.keys()))
        self.setup_feature_extractor()

        # self.args = args
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.setup_data()
        # self.setup_model()
        # self.setup_optimizer()
        # self.setup_criterion()
        # self.setup_strategies()
        # self.setup_agents()
        # self.curriculum_stage = 0.0
        # self.last_accuracy = 0.0
        # self.metrics = []
        # self.start_time = time.time()

    def setup_feature_extractor(self):
        # Create a new model for feature extraction
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor.eval()  # Set to evaluation mode

    def extract_features(self, indices):
        self.feature_extractor.eval()
        features = []
        subset = Subset(self.full_dataset, indices)
        dataloader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=self.args.workers)
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                feature = self.feature_extractor(inputs)
                features.append(feature.squeeze().cpu().numpy())
        
        return np.vstack(features)

    def setup_data(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        self.test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        
        self.current_indices = self.smart_initial_sampling(self.args.initial_sample_size)
        self.update_train_loader()
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.workers)

    def setup_model(self):
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        self.model = self.model.to(self.device)
    def get_weights(self):
        return self.meta_agent.get_weights()


    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)

    def setup_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def setup_strategies(self):
        self.strategies = {
        "S_U": self.get_uncertain_samples,
        "S_C": self.get_class_balanced_samples,
        "S_B": self.get_difficult_samples,  # Changed from get_diverse_samples
        "S_D": self.get_diverse_samples
    }
        
    def calculate_strategy_rewards(self):
         rewards = {}
         for strategy in self.strategies:
            # 1. Performance improvement
            current_accuracy = self.last_accuracy
            previous_accuracy = self.metrics[-2]['test_acc'] if len(self.metrics) > 1 else 0
            accuracy_improvement = max(0, current_accuracy - previous_accuracy)

            # 2. Sample diversity
            new_samples = list(set(self.current_indices) - set(self.previous_indices))
            diversity_score = self.calculate_sample_diversity(new_samples)

            # 3. Class balance improvement
            current_balance = self.calculate_class_balance(self.current_indices)
            previous_balance = self.calculate_class_balance(self.previous_indices)
            balance_improvement = max(0, current_balance - previous_balance)

            # 4. Difficulty progression
            difficulty_scores = self.assess_sample_difficulty(new_samples)
            difficulty_score = np.mean(difficulty_scores)

            # 5. Strategy utilization
            strategy_usage = self.meta_agent.weights[strategy]

            # Combine factors with weights
            reward = (
                0.4 * accuracy_improvement +
                0.2 * diversity_score +
                0.2 * balance_improvement +
                0.1 * difficulty_score +
                0.1 * strategy_usage
            )

            rewards[strategy] = reward

         # Normalize rewards
         min_reward = min(rewards.values())
         max_reward = max(rewards.values())
         if max_reward > min_reward:
            rewards = {s: (r - min_reward) / (max_reward - min_reward) for s, r in rewards.items()}
         else:
             rewards = {s: 1.0 for s in self.strategies}  # If all rewards are equal

         return rewards


    

    def setup_agents(self):
        self.selection_agents = {
            "S_U": SelectionAgent("uncertainty", self.args.organization, self.args.project),
            "S_C": SelectionAgent("class balance", self.args.organization, self.args.project),
            "S_B": SelectionAgent("boundary", self.args.organization, self.args.project),
            "S_D": SelectionAgent("diversity", self.args.organization, self.args.project)
        }
        self.moderator = ModeratorAgent(self.args.organization, self.args.project)
        self.meta_agent = MetaAgent(list(self.strategies.keys()))

    def update_train_loader(self):
        subset = Subset(self.full_dataset, self.current_indices)
        self.train_loader = DataLoader(subset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.workers)

    def smart_initial_sampling(self, n_samples):
        return np.random.choice(len(self.full_dataset), n_samples, replace=False)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # Check if the batch size is greater than 1
            if inputs.size(0) <= 1:
                print(f"Skipping batch {batch_idx} with size {inputs.size(0)}")
                continue
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % self.args.print_freq == 0:
                print(f'Epoch: {self.current_epoch} [{batch_idx * len(inputs)}/{len(self.train_loader.dataset)} ({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        return total_loss / len(self.train_loader), correct / total

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        accuracy = correct / total
        f1 = f1_score(all_targets, all_preds, average='weighted')
        precision = precision_score(all_targets, all_preds, average='weighted')
        recall = recall_score(all_targets, all_preds, average='weighted')
        conf_matrix = confusion_matrix(all_targets, all_preds)

        return total_loss / len(self.test_loader), accuracy, f1, precision, recall, conf_matrix

    def assess_sample_difficulty(self, samples):
        self.model.eval()
        difficulties = []
        dataloader = DataLoader(Subset(self.full_dataset, samples), batch_size=64, shuffle=False, num_workers=self.args.workers)
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                
                # Method 1: Entropy of predictions
                probabilities = F.softmax(outputs, dim=1)
                entropies = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
                
                # Method 2: 1 - max probability
                max_probs = torch.max(probabilities, dim=1)[0]
                uncertainty = 1 - max_probs
                
                # Method 3: Margin of top two classes
                top2_probs, _ = torch.topk(probabilities, k=2, dim=1)
                margins = top2_probs[:, 0] - top2_probs[:, 1]
                
                # Method 4: Loss value
                losses = F.cross_entropy(outputs, targets, reduction='none')
                
                # Combine methods (you can adjust these weights)
                difficulty = (
                    0.3 * entropies.cpu().numpy() +
                    0.3 * uncertainty.cpu().numpy() +
                    0.2 * (1 - margins.cpu().numpy()) +  # Invert margin so higher value = more difficult
                    0.2 * losses.cpu().numpy()
                )
                
                difficulties.extend(difficulty)

        # Normalize difficulties to [0, 1] range
        difficulties = np.array(difficulties)
        difficulties = (difficulties - difficulties.min()) / (difficulties.max() - difficulties.min() + 1e-10)
        
        return difficulties
    
    def calculate_class_balance(self, indices):
        class_counts = Counter([self.full_dataset[i][1] for i in indices])
        total = sum(class_counts.values())
        class_proportions = [count / total for count in class_counts.values()]
        return 1 - np.std(class_proportions)  # Higher value means better balance
    
    def get_difficult_samples(self, n_samples):
        current_subset = Subset(self.full_dataset, self.current_indices)
        
        # If we don't have enough samples in our current subset, return all indices
        if len(current_subset) <= n_samples:
            return list(range(len(current_subset)))

        difficulties = self.assess_sample_difficulty(range(len(current_subset)))
        
        # Select the n_samples most difficult samples
        difficult_indices = np.argsort(difficulties)[-n_samples:]
        
        return difficult_indices

    def get_uncertain_samples(self, n_samples):
        self.model.eval()
        uncertainties = []
        with torch.no_grad():
            for inputs, _ in DataLoader(self.full_dataset, batch_size=self.args.batch_size):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                uncertainties.extend((-probabilities * torch.log(probabilities)).sum(1).cpu().numpy())
        
        uncertain_indices = np.argsort(uncertainties)[::-1]
        return [idx for idx in uncertain_indices if idx not in self.current_indices][:n_samples]

    def get_class_balanced_samples(self, n_samples):
        class_counts = Counter([self.full_dataset[i][1] for i in self.current_indices])
        samples_per_class = n_samples // 10
        balanced_samples = []
        for class_label in range(10):
            class_samples = [i for i in range(len(self.full_dataset)) if self.full_dataset[i][1] == class_label and i not in self.current_indices]
            balanced_samples.extend(np.random.choice(class_samples, min(samples_per_class, len(class_samples)), replace=False))
        return balanced_samples
        
    


    def get_diverse_samples(self, n_samples):
         n_samples = int(n_samples)
        
         if len(self.current_indices) <= n_samples:
            return self.current_indices

         try:
            features = self.extract_features(self.current_indices)
         except Exception as e:
            print(f"Error in feature extraction: {e}")
            # Fallback to random sampling if feature extraction fails
            return np.random.choice(self.current_indices, size=n_samples, replace=False)

         if features.shape[0] == 0:
            print("No features extracted. Falling back to random sampling.")
            return np.random.choice(self.current_indices, size=n_samples, replace=False)

         scaler = StandardScaler()
         normalized_features = scaler.fit_transform(features)

         n_clusters = min(n_samples, len(self.current_indices))
         self.kmeans = KMeans(n_clusters=n_clusters, n_init=10)
         cluster_labels = self.kmeans.fit_predict(normalized_features)

         diverse_samples = []
         for cluster in range(n_clusters):
             cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
             if cluster_indices:
                cluster_center = self.kmeans.cluster_centers_[cluster]
                distances = [np.linalg.norm(normalized_features[i] - cluster_center) for i in cluster_indices]
                closest_index = cluster_indices[np.argmin(distances)]
                diverse_samples.append(self.current_indices[closest_index])

         if len(diverse_samples) < n_samples:
            remaining = list(set(self.current_indices) - set(diverse_samples))
            diverse_samples.extend(np.random.choice(remaining, n_samples - len(diverse_samples), replace=False))

         return diverse_samples

    def calculate_sample_diversity(self, new_samples):
        if not new_samples:
            return 0

        if self.kmeans is None:
            # If KMeans hasn't been run yet, we can't calculate diversity meaningfully
            return 0

        features = self.extract_features(new_samples)
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)

        # Assign new samples to existing clusters
        cluster_assignments = self.kmeans.predict(normalized_features)

        # Calculate the distribution of new samples across clusters
        cluster_distribution = np.bincount(cluster_assignments, minlength=self.kmeans.n_clusters)
        cluster_distribution = cluster_distribution / len(new_samples)

        # Calculate entropy as a measure of diversity
        entropy = -np.sum(cluster_distribution * np.log(cluster_distribution + 1e-10))

        # Normalize entropy to [0, 1]
        max_entropy = np.log(self.kmeans.n_clusters)
        normalized_entropy = entropy / max_entropy

        return normalized_entropy




    def write_intermediate_statistics(self):
        metrics_file = f"{self.args.save_path}/metrics.csv"
        file_exists = os.path.isfile(metrics_file)
        
        with open(metrics_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.metrics[-1].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(self.metrics[-1])

        # Save current model state
        torch.save(self.model.state_dict(), f"{self.args.save_path}/model_epoch_{self.current_epoch}.pth")

        # Save current weights
        weights = self.get_weights()
        with open(f"{self.args.save_path}/weights_epoch_{self.current_epoch}.json", 'w') as f:
            json.dump(weights, f)

        # Save current indices
        np.save(f"{self.args.save_path}/indices_epoch_{self.current_epoch}.npy", self.current_indices)

        print(f"Intermediate statistics saved for epoch {self.current_epoch}")

    def get_misclassified_samples(self, n_samples):
        self.model.eval()
        misclassified = []
        with torch.no_grad():
            for inputs, targets in DataLoader(self.full_dataset, batch_size=self.args.batch_size):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                misclassified.extend((predicted != targets).nonzero().squeeze().cpu().numpy())
        return np.random.choice(misclassified, min(n_samples, len(misclassified)), replace=False)

   
    def get_curriculum_samples(self, n_samples):
        class_distribution = Counter([self.full_dataset[i][1] for i in self.current_indices])
        
        # Get proposals from all agents
        agent_proposals = {}
        for strategy, agent in self.selection_agents.items():
            proposal = agent.propose_selection(self.last_accuracy, self.curriculum_stage, class_distribution)
            agent_proposals[strategy] = proposal

        # Let the moderator suggest weight adjustments
        current_weights = self.meta_agent.get_weights()
        suggested_weights, explanation = self.moderator.moderate_discussion(agent_proposals, self.last_accuracy, self.curriculum_stage, current_weights)

        # Calculate rewards using the new method
        rewards = self.calculate_strategy_rewards()

        # Update meta agent weights
        self.meta_agent.update_weights_with_moderator(rewards, suggested_weights)

        # Get final weights from meta agent
        final_weights = self.meta_agent.get_weights()

        # Log the proposals, moderator's decision, and final weights
        print("Agent Proposals:")
        for strategy, proposal in agent_proposals.items():
            print(f"{strategy}: {proposal}")
        print(f"Moderator Suggested Weights: {suggested_weights}")
        print(f"Moderator Explanation: {explanation}")
        print(f"Calculated Rewards: {rewards}")
        print(f"Final Weights: {final_weights}")

        all_samples = []
        for strategy, weight in final_weights.items():
            n_strategy_samples = int(n_samples * weight)
            strategy_samples = self.strategies[strategy](n_strategy_samples)
            all_samples.extend(strategy_samples)
        
        if len(all_samples) < n_samples:
            remaining = n_samples - len(all_samples)
            available_indices = list(set(range(len(self.full_dataset))) - set(all_samples) - set(self.current_indices))
            additional_samples = np.random.choice(available_indices, size=remaining, replace=False)
            all_samples.extend(additional_samples)
        
        return all_samples[:n_samples]

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.current_indices = checkpoint['current_indices']
        self.curriculum_stage = checkpoint['curriculum_stage']
        self.meta_agent.weights = checkpoint['meta_agent_weights']
        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train(self):
        best_accuracy = 0
        for self.current_epoch in range(self.args.epochs):
            epoch_start_time = time.time()
            
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc, f1, precision, recall, conf_matrix = self.evaluate()
            self.scheduler.step()

            self.last_accuracy = test_acc
            self.curriculum_stage = min(1.0, (self.current_epoch + 1) / self.args.epochs)

            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - self.start_time

            metrics = {
                "epoch": self.current_epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "dataset_size": len(self.current_indices),
                "curriculum_stage": self.curriculum_stage,
                "epoch_time": epoch_time,
                "total_time": total_time
            }
            self.metrics.append(metrics)

            print(f"Epoch {self.current_epoch+1}/{self.args.epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
                  f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
                  f"Epoch Time: {epoch_time:.2f}s, Total Time: {total_time:.2f}s")

            if test_acc > best_accuracy:
                best_accuracy = test_acc
                self.save_checkpoint({
                    'epoch': self.current_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_accuracy': best_accuracy,
                    'current_indices': self.current_indices,
                    'curriculum_stage': self.curriculum_stage,
                    'meta_agent_weights': self.meta_agent.get_weights()
                }, f"{self.args.save_path}/best_model.pth")

            if len(self.current_indices) < self.args.max_budget:
                new_samples = self.get_curriculum_samples(min(self.args.samples_per_epoch, self.args.max_budget - len(self.current_indices)))
                self.current_indices = np.unique(np.concatenate([self.current_indices, new_samples]))
                self.update_train_loader()
             # Write intermediate statistics every N epochs
            if (self.current_epoch + 1) % self.args.save_interval == 0:
                self.write_intermediate_statistics()


        self.save_final_statistics()

    def save_final_statistics(self):
        # Save metrics to CSV
        with open(f"{self.args.save_path}/metrics.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.metrics[0].keys())
            writer.writeheader()
            for metric in self.metrics:
                writer.writerow(metric)

        # Save confusion matrix
        _, _, _, _, _, conf_matrix = self.evaluate()
        np.savetxt(f"{self.args.save_path}/confusion_matrix.csv", conf_matrix, delimiter=",")

        # Save final model state
        torch.save(self.model.state_dict(), f"{self.args.save_path}/final_model.pth")

        # Save meta-agent weights
        with open(f"{self.args.save_path}/meta_agent_weights.json", 'w') as f:
            json.dump(self.meta_agent.get_weights(), f)

        # Save training arguments
        with open(f"{self.args.save_path}/training_args.json", 'w') as f:
            json.dump(vars(self.args), f, indent=4)

        print(f"Final statistics saved to {self.args.save_path}")

    def run(self):
        if self.args.resume:
            self.load_checkpoint(self.args.resume)
        self.train()
        print("Training completed.")

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Curriculum Learning')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--max-budget', type=int, default=25000, help='maximum number of samples to use (default: 25000)')
    parser.add_argument('--initial-sample-size', type=int, default=1000, help='initial number of samples (default: 1000)')
    parser.add_argument('--samples-per-epoch', type=int, default=1000, help='number of samples to add per epoch (default: 1000)')
    parser.add_argument('--save-path', type=str, default='./checkpoints_message_cifar10_', help='path to save checkpoints')
    parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint (default: none)')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency (default: 10)')
    parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
    parser.add_argument('--save-interval', type=int, default=10, help='save intermediate results every N epochs (default: 10)')
    parser.add_argument('--organization', type=str, default='org-K8Ym1ORBOhkM4bMusetuJ95u', help='OpenAI organization ID')
    parser.add_argument('--project', type=str, default='proj_EArB7IlQvNIkrENaKjCVItBE', help='OpenAI project ID')
    #parser.add_argument('--openai-api-key', type=str, required=True, help='OpenAI API key for agent communication')
    
    args = parser.parse_args()
    
    args.save_path = f"{args.save_path}{args.max_budget}"
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    trainer = CurriculumCIFAR10Trainer(args)
    trainer.run()

if __name__ == '__main__':
    main()

    def get_misclassified_samples(self, n_samples):
        self.model.eval()
        misclassified = []
        with torch.no_grad():
            for inputs, targets in DataLoader(self.full_dataset, batch_size=self.args.batch_size):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                misclassified.extend((predicted != targets).nonzero().squeeze().cpu().numpy())
        return np.random.choice(misclassified, min(n_samples, len(misclassified)), replace=False)

   
    def get_curriculum_samples(self, n_samples):
        class_distribution = Counter([self.full_dataset[i][1] for i in self.current_indices])
        
        # Get proposals from all agents
        agent_proposals = {}
        for strategy, agent in self.selection_agents.items():
            proposal = agent.propose_selection(self.last_accuracy, self.curriculum_stage, class_distribution)
            agent_proposals[strategy] = proposal

        # Let the moderator suggest weight adjustments
        current_weights = self.meta_agent.get_weights()
        suggested_weights, explanation = self.moderator.moderate_discussion(agent_proposals, self.last_accuracy, self.curriculum_stage, current_weights)

        # Calculate rewards using the new method
        rewards = self.calculate_strategy_rewards()

        # Update meta agent weights
        self.meta_agent.update_weights_with_moderator(rewards, suggested_weights)

        # Get final weights from meta agent
        final_weights = self.meta_agent.get_weights()

        # Log the proposals, moderator's decision, and final weights
        print("Agent Proposals:")
        for strategy, proposal in agent_proposals.items():
            print(f"{strategy}: {proposal}")
        print(f"Moderator Suggested Weights: {suggested_weights}")
        print(f"Moderator Explanation: {explanation}")
        print(f"Calculated Rewards: {rewards}")
        print(f"Final Weights: {final_weights}")

        all_samples = []
        for strategy, weight in final_weights.items():
            n_strategy_samples = int(n_samples * weight)
            strategy_samples = self.strategies[strategy](n_strategy_samples)
            all_samples.extend(strategy_samples)
        
        if len(all_samples) < n_samples:
            remaining = n_samples - len(all_samples)
            available_indices = list(set(range(len(self.full_dataset))) - set(all_samples) - set(self.current_indices))
            additional_samples = np.random.choice(available_indices, size=remaining, replace=False)
            all_samples.extend(additional_samples)
        
        return all_samples[:n_samples]

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.current_indices = checkpoint['current_indices']
        self.curriculum_stage = checkpoint['curriculum_stage']
        self.meta_agent.weights = checkpoint['meta_agent_weights']
        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train(self):
        best_accuracy = 0
        for self.current_epoch in range(self.args.epochs):
            epoch_start_time = time.time()
            
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc, f1, precision, recall, conf_matrix = self.evaluate()
            self.scheduler.step()

            self.last_accuracy = test_acc
            self.curriculum_stage = min(1.0, (self.current_epoch + 1) / self.args.epochs)

            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - self.start_time

            metrics = {
                "epoch": self.current_epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "dataset_size": len(self.current_indices),
                "curriculum_stage": self.curriculum_stage,
                "epoch_time": epoch_time,
                "total_time": total_time
            }
            self.metrics.append(metrics)

            print(f"Epoch {self.current_epoch+1}/{self.args.epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
                  f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
                  f"Epoch Time: {epoch_time:.2f}s, Total Time: {total_time:.2f}s")

            if test_acc > best_accuracy:
                best_accuracy = test_acc
                self.save_checkpoint({
                    'epoch': self.current_epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_accuracy': best_accuracy,
                    'current_indices': self.current_indices,
                    'curriculum_stage': self.curriculum_stage,
                    'meta_agent_weights': self.meta_agent.get_weights()
                }, f"{self.args.save_path}/best_model.pth")

            if len(self.current_indices) < self.args.max_budget:
                new_samples = self.get_curriculum_samples(min(self.args.samples_per_epoch, self.args.max_budget - len(self.current_indices)))
                self.current_indices = np.unique(np.concatenate([self.current_indices, new_samples]))
                self.update_train_loader()
             # Write intermediate statistics every N epochs
            if (self.current_epoch + 1) % self.args.save_interval == 0:
                self.write_intermediate_statistics()


        self.save_final_statistics()

    def save_final_statistics(self):
        # Save metrics to CSV
        with open(f"{self.args.save_path}/metrics.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.metrics[0].keys())
            writer.writeheader()
            for metric in self.metrics:
                writer.writerow(metric)

        # Save confusion matrix
        _, _, _, _, _, conf_matrix = self.evaluate()
        np.savetxt(f"{self.args.save_path}/confusion_matrix.csv", conf_matrix, delimiter=",")

        # Save final model state
        torch.save(self.model.state_dict(), f"{self.args.save_path}/final_model.pth")

        # Save meta-agent weights
        with open(f"{self.args.save_path}/meta_agent_weights.json", 'w') as f:
            json.dump(self.meta_agent.get_weights(), f)

        # Save training arguments
        with open(f"{self.args.save_path}/training_args.json", 'w') as f:
            json.dump(vars(self.args), f, indent=4)

        print(f"Final statistics saved to {self.args.save_path}")

    def run(self):
        if self.args.resume:
            self.load_checkpoint(self.args.resume)
        self.train()
        print("Training completed.")

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Curriculum Learning')
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--max-budget', type=int, default=10000, help='maximum number of samples to use (default: 25000)')
    parser.add_argument('--initial-sample-size', type=int, default=1000, help='initial number of samples (default: 1000)')
    parser.add_argument('--samples-per-epoch', type=int, default=1000, help='number of samples to add per epoch (default: 1000)')
    parser.add_argument('--save-path', type=str, default='./checkpoints_message_cifar10_', help='path to save checkpoints')
    parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint (default: none)')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency (default: 10)')
    parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
    parser.add_argument('--save-interval', type=int, default=10, help='save intermediate results every N epochs (default: 10)')
    parser.add_argument('--organization', type=str, default='org-K8Ym1ORBOhkM4bMusetuJ95u', help='OpenAI organization ID')
    parser.add_argument('--project', type=str, default='proj_EArB7IlQvNIkrENaKjCVItBE', help='OpenAI project ID')
    #parser.add_argument('--openai-api-key', type=str, required=True, help='OpenAI API key for agent communication')
    
    args = parser.parse_args()
    
    args.save_path = f"{args.save_path}{args.max_budget}"
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    trainer = CurriculumCIFAR10Trainer(args)
    trainer.run()

if __name__ == '__main__':
    main()