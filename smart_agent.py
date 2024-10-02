import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from typing import List, Tuple, Dict
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score
import json
import random
import asyncio
from functools import lru_cache
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

from torch.cuda.amp import autocast, GradScaler

import cProfile
import pstats
import io
from pstats import SortKey



import math
import numpy as np

class MetaAgent:
    def __init__(self, llm_strategies, baseline_strategies, initial_temperature=1.0, cooling_rate=0.995):
        self.llm_strategies = llm_strategies
        self.baseline_strategies = baseline_strategies
        self.all_strategies = {**llm_strategies, **baseline_strategies, "LLM_multi_agent": None}
        
        self.strategy_weights = {s: 1.0 / len(self.all_strategies) for s in self.all_strategies}
        self.performance_history = {s: [] for s in self.all_strategies}
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def update_performance(self, strategy, performance):
        self.performance_history[strategy].append(performance)
        if len(self.performance_history[strategy]) > 10:
            self.performance_history[strategy].pop(0)

    def get_strategy_weights(self):
        avg_performances = {s: np.mean(perf) if perf else 0 for s, perf in self.performance_history.items()}
        exp_performances = {s: math.exp(p / self.temperature) for s, p in avg_performances.items()}
        total_exp = sum(exp_performances.values())
        new_weights = {s: ep / total_exp for s, ep in exp_performances.items()}

        exploration_factor = 0.1
        for s in self.all_strategies:
            self.strategy_weights[s] = (1 - exploration_factor) * new_weights[s] + exploration_factor / len(self.all_strategies)

        return self.strategy_weights

    def cool_temperature(self):
        self.temperature *= self.cooling_rate

    def select_strategy(self):
        weights = self.get_strategy_weights()
        strategies = list(weights.keys())
        probabilities = list(weights.values())
        return np.random.choice(strategies, p=probabilities)

    def get_llm_strategy_weights(self):
        llm_weights = {s: self.strategy_weights[s] for s in self.llm_strategies}
        total_weight = sum(llm_weights.values())
        return {s: w / total_weight for s, w in llm_weights.items()}

    
class InteractiveCIFAR10Trainer:
    def __init__(self, args, api_key, budget_percent=0.1, initial_sample_size=1000, misclassification_memory=5):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.api_key = api_key
        self.misclassified_samples = {}
        self.misclassification_memory = misclassification_memory
        self.initial_sample_size=initial_sample_size

        self.feature_extractor = nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
        self.feature_extractor = self.feature_extractor.to(self.device)
        # Cache for features
        self.cached_features = None
        
        self.feature_extractor.eval()
        
        # Data loading and preprocessing
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        # Budget setup
        self.max_budget = int(len(self.full_dataset) * budget_percent)
        self.current_budget = initial_sample_size
        print(f"Max budget: {self.max_budget} samples")
        
        # Model setup
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        self.model = self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        
        # Smart initial sampling
        self.current_indices=[]
        self.current_indices = self.smart_initial_sampling(initial_sample_size)
        self.update_train_loader()
        
        self.curriculum_stage = 0.0
        self.last_accuracy = 0.0

        # Mixed precision setup
        self.scaler = GradScaler()

        self.llm_strategies = {
            "S_B": self.get_class_balanced_samples,
            "S_U": self.get_uncertain_samples,
            "S_D": self.get_diverse_samples,
            "S_M": self.get_misclassified_samples
        }

        self.baseline_strategies = {
            "random": self.get_random_samples,
            "herding": self.get_herding_samples
        }

        self.all_strategies = {
            "LLM_multi_agent": self.get_curriculum_samples,
            **self.baseline_strategies
        }

        self.strategy_performances = {strategy: [] for strategy in self.all_strategies}
        self.strategy_sample_counts = {strategy: 0 for strategy in self.all_strategies}

        # LLM setup
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        self.chat_model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m").to(self.device)

        # Regularization parameters
        self.weight_decay = args.weight_decay
        self.dropout_rate = args.dropout_rate

        self.meta_agent = MetaAgent(self.llm_strategies, self.baseline_strategies)

        
    @lru_cache(maxsize=100)
    def get_model_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.chat_model.generate(**inputs, max_new_tokens=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_random_samples(self, n_samples: int) -> List[int]:
        unselected_indices = list(set(range(len(self.full_dataset))) - set(self.current_indices))
        return np.random.choice(unselected_indices, min(n_samples, len(unselected_indices)), replace=False).tolist()

    def get_herding_samples(self, n_samples: int) -> List[int]:
        features = self.extract_features(self.full_dataset)
        unselected_indices = list(set(range(len(self.full_dataset))) - set(self.current_indices))
        unselected_features = features[unselected_indices]
        
        mean_feature = np.mean(unselected_features, axis=0)
        selected_indices = []
        
        for _ in range(n_samples):
            if len(unselected_indices) == 0:
                break
            distances = cdist(unselected_features, [mean_feature])
            idx = np.argmin(distances)
            selected_indices.append(unselected_indices[idx])
            unselected_indices.pop(idx)
            unselected_features = np.delete(unselected_features, idx, axis=0)
        
        return selected_indices

    def get_curriculum_samples(self, n_samples: int) -> List[int]:
        strategy = self.meta_agent.select_strategy()
        
        if strategy == "LLM_multi_agent":
            class_distribution = self.get_class_distribution()
            agent_proposals = self.get_agent_proposals(class_distribution)
            consensus = self.agent_interaction(agent_proposals)
            
            print("\nAgent Proposals:")
            for strategy, proposal in agent_proposals.items():
                print(f"{strategy}: {proposal}")
            print(f"\nConsensus: {consensus}")
            
            weights = self.meta_agent.get_llm_strategy_weights()
            
            selected_samples = []
            for sub_strategy, weight in weights.items():
                n_strategy_samples = max(1, int(n_samples * weight))
                samples = self.llm_strategies[sub_strategy](n_strategy_samples)
                selected_samples.extend(samples)
            
            return selected_samples[:n_samples]
        else:
            return self.all_strategies[strategy](n_samples)
    
    def get_misclassified_samples(self, n_samples: int) -> List[int]:
        misclassified = list(self.misclassified_samples.keys())
        if len(misclassified) > n_samples:
            return np.random.choice(misclassified, n_samples, replace=False).tolist()
        else:
            return misclassified
        

    
    def get_diverse_samples(self, n_samples: int) -> List[int]:
        features = self.extract_features(self.full_dataset)
        unselected_indices = list(set(range(len(self.full_dataset))) - set(self.current_indices))
        unselected_features = features[unselected_indices]
        
        kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(unselected_features)
        
        diverse_samples = []
        for i in range(n_samples):
            cluster_points = np.where(cluster_labels == i)[0]
            if len(cluster_points) > 0:
                sample_index = np.random.choice(cluster_points)
                diverse_samples.append(unselected_indices[sample_index])
        
        return diverse_samples
   

    def get_class_distribution(self):
        class_counts = {cls: 0 for cls in self.full_dataset.classes}
        for idx in self.current_indices:
            _, label = self.full_dataset[idx]
            class_counts[self.full_dataset.classes[label]] += 1
        return class_counts
    
    def get_certain_samples(self, n_samples: int, model=None) -> List[int]:
        if model is None:
            model = self.model
    
        model.eval()
        certainties = []
        unselected_indices = list(set(range(len(self.full_dataset))) - set(self.current_indices))
        unselected_subset = Subset(self.full_dataset, unselected_indices)
        unselected_loader = DataLoader(unselected_subset, batch_size=100, shuffle=False, num_workers=2)
    
        with torch.no_grad():
            for inputs, _ in unselected_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                certainties.extend(torch.max(probabilities, dim=1)[0].cpu().numpy())
    
        certain_indices = np.argsort(certainties)[-n_samples:]
        return [unselected_indices[i] for i in certain_indices]


    def get_uncertain_samples(self, n_samples: int, model=None) -> List[int]:
        if model is None:
            model = self.model
    
        model.eval()
        uncertainties = []
        unselected_indices = list(set(range(len(self.full_dataset))) - set(self.current_indices))
        unselected_subset = Subset(self.full_dataset, unselected_indices)
        unselected_loader = DataLoader(unselected_subset, batch_size=100, shuffle=False, num_workers=2)
    
        with torch.no_grad():
            for inputs, _ in unselected_loader:
                inputs = inputs.to(self.device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                uncertainties.extend(1 - torch.max(probabilities, dim=1)[0].cpu().numpy())
    
        uncertain_indices = np.argsort(uncertainties)[-n_samples:]
        return [unselected_indices[i] for i in uncertain_indices]
    
    

    def get_class_balanced_samples(self, n_samples: int) -> List[int]:
        class_counts = [0] * 10
        for idx in range(len(self.full_dataset)):
            _, label = self.full_dataset[idx]
            class_counts[label] += 1
        
        samples_per_class = n_samples // 10
        new_indices = []
        for class_label in range(10):
            class_samples = [idx for idx in range(len(self.full_dataset)) 
                             if self.full_dataset[idx][1] == class_label and idx not in self.current_indices]
            samples_needed = max(0, samples_per_class - class_counts[class_label])
            new_indices.extend(np.random.choice(class_samples, min(samples_needed, len(class_samples)), replace=False))
        
        return new_indices

    @torch.no_grad() 
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in DataLoader(self.test_dataset, batch_size=256, shuffle=False, num_workers=2):
                inputs, targets = inputs.to(self.device,non_blocking=True), targets.to(self.device,non_blocking=True)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        accuracy = correct / total
        f1 = f1_score(all_targets, all_preds, average='weighted')
        precision = precision_score(all_targets, all_preds, average='weighted')
        recall = recall_score(all_targets, all_preds, average='weighted')
        
        return accuracy, f1, precision, recall, all_preds, all_targets



    def update_misclassified_samples(self, all_preds, all_targets,current_epoch):
        misclassified_indices = np.where(np.array(all_preds) != np.array(all_targets))[0]
        
        
        for idx in misclassified_indices:
            if idx in self.misclassified_samples:
                self.misclassified_samples[idx].append(current_epoch)
            else:
                self.misclassified_samples[idx] = [current_epoch]
        
        self.misclassified_samples = {
            k: v for k, v in self.misclassified_samples.items() 
            if current_epoch - v[-1] <= self.misclassification_memory
        }
        
        print(f"Number of misclassified samples in this epoch: {len(misclassified_indices)}")
        print(f"Total number of samples in misclassification memory: {len(self.misclassified_samples)}")
        
        persistent_misclassifications = [idx for idx, epochs in self.misclassified_samples.items() if len(epochs) > 1]
        print(f"Number of samples misclassified in multiple epochs: {len(persistent_misclassifications)}")
        
        class_distribution = Counter([self.test_dataset[idx][1] for idx in misclassified_indices])
        print("Misclassifications by class:")
        for class_idx, count in class_distribution.items():
            print(f"Class {self.test_dataset.classes[class_idx]}: {count} misclassifications")
        
        # Print examples of misclassifications
        print("\nExamples of misclassifications:")
        for idx in misclassified_indices[:5]:  # Print first 5 misclassifications
            true_label = self.test_dataset.classes[all_targets[idx]]
            pred_label = self.test_dataset.classes[all_preds[idx]]
            print(f"Sample {idx}: True label: {true_label}, Predicted: {pred_label}")
    
    @torch.no_grad()
    def extract_features(self, dataset):
        if self.cached_features is not None:
            return self.cached_features

        
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)
        features = []

        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device,non_blocking=True)
                outputs = self.feature_extractor(inputs)
                features.append(outputs.squeeze().cpu().numpy())
        self.cached_features=np.vstack(features)
        return self.cached_features
    
    def train_small_model(self, subset, epochs=5):
        model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)
        model = model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        loader = DataLoader(subset, batch_size=64, shuffle=True, num_workers=2)
        
        for epoch in range(epochs):
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        return model

    def update_train_loader(self):
        subset = Subset(self.full_dataset, self.current_indices)
        self.train_loader = DataLoader(subset, batch_size=128, shuffle=True, num_workers=2)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            
            # Mixed precision training
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(self.train_loader), correct / total

    def smart_initial_sampling(self, n_samples):
        print("Performing smart initial sampling...")

        sample_indices = np.random.choice(len(self.full_dataset), size=5000, replace=False)
        sampled_dataset = Subset(self.full_dataset, sample_indices)
        features = self.extract_features(sampled_dataset)
        
        kmeans = KMeans(n_clusters=n_samples // 4, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        diverse_samples = [np.where(cluster_labels == i)[0][0] for i in range(kmeans.n_clusters)]

        # 2. Class-balanced sampling
        samples_per_class = n_samples // (4 * 10)  # 10 classes in CIFAR-10
        balanced_samples = self.get_class_balanced_samples(samples_per_class)

        # 3. Uncertainty-based sampling
        subset_indices = np.random.choice(len(self.full_dataset), 5000, replace=False)
        subset = Subset(self.full_dataset, subset_indices)
        small_model = self.train_small_model(subset)
        uncertain_samples = self.get_uncertain_samples(n_samples // 4, model=small_model)

        # 4. Certainty-based sampling (most confident predictions)
        certain_samples = self.get_certain_samples(n_samples // 4, model=small_model)

        # Combine all sampling methods
        initial_samples = list(set(diverse_samples + balanced_samples + uncertain_samples + certain_samples))

        # If we need more samples, add random samples
        if len(initial_samples) < n_samples:
            remaining_samples = n_samples - len(initial_samples)
            random_samples = np.random.choice(
                list(set(range(len(self.full_dataset))) - set(initial_samples)),
                remaining_samples,
                replace=False
            )
            initial_samples.extend(random_samples)

        print(f"Initial sample composition:")
        print(f"Diverse samples: {len(diverse_samples)}")
        print(f"Balanced samples: {len(balanced_samples)}")
        print(f"Uncertain samples: {len(uncertain_samples)}")
        print(f"Certain samples: {len(certain_samples)}")
        print(f"Random samples: {len(initial_samples) - len(diverse_samples) - len(balanced_samples) - len(uncertain_samples) - len(certain_samples)}")
        print(f"Total unique samples: {len(initial_samples)}")

        return initial_samples
    
    def get_agent_proposals(self, class_distribution):
        proposals = {}
        for strategy, strategy_func in self.llm_strategies.items():
            prompt = f"""
            As the {strategy} agent for CIFAR-10 image classification:
            Current model accuracy: {self.last_accuracy:.2f}
            Current curriculum stage: {self.curriculum_stage:.2f}
            Class distribution: {class_distribution}
            Current budget: {len(self.current_indices)} / {self.max_budget}
            Misclassified samples: {len(self.misclassified_samples)}

            Propose how to select the next batch of samples and why. Consider:
            1. The strengths of your strategy ({strategy})
            2. The current model performance and curriculum stage
            3. The current class distribution and budget constraints
            4. The misclassification information

            Respond in 50 words or less.
            """
            proposals[strategy] = self.get_model_response(prompt)
        return proposals
    

    def agent_interaction(self, proposals):
        prompt = f"""
        Agent Proposals:
        {' '.join([f'{strategy}: {proposal}' for strategy, proposal in proposals.items()])}

        As the moderator, facilitate a discussion between the agents. They should:
        1. Critique each other's proposals
        2. Suggest improvements or compromises
        3. Come to a consensus on the best approach

        Provide a summary of their discussion and the final consensus in 100 words or less.
        """
        
        return self.get_model_response(prompt)
    
    
    def setup_model(self):
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(self.args.dropout_rate),
            nn.Linear(num_ftrs, 10)
        )
        self.model = self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)


    def parse_consensus(self, consensus):
        weights = {strategy: 0.25 for strategy in self.llm_strategies}
        
        for strategy in self.llm_strategies:
            if strategy.lower() in consensus.lower():
                weights[strategy] += 0.1
        
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

    def train_with_strategy(self, strategy: str, budget_percent: float, epochs: int):
        profiler=cProfile.Profile
        profiler.enable()
        self.current_indices = self.smart_initial_sampling(self.initial_sample_size)
        self.max_budget = int(len(self.full_dataset) * budget_percent)
        self.strategy_performances[strategy] = []
        self.strategy_sample_counts[strategy] = len(self.current_indices)

        self.setup_model()  # Reset the model for each strategy
        self.update_train_loader()

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            test_acc, f1, precision, recall, all_preds, all_targets = self.evaluate()
            
            self.meta_agent.update_performance(strategy, test_acc)
            self.meta_agent.cool_temperature()
            self.last_accuracy = test_acc
            self.curriculum_stage = min(test_acc, 0.9)

            self.strategy_performances[strategy].append(test_acc)
            
            print(f"Strategy: {strategy}, Budget: {budget_percent:.2f}, Epoch: {epoch+1}/{epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
            
            self.update_misclassified_samples(all_preds, all_targets,epoch)
            
            # Dynamic sample addition
            if self.should_add_samples(epoch, test_acc):
                new_samples = self.all_strategies[strategy](min(200, self.max_budget - len(self.current_indices)))
                self.add_new_samples(new_samples)
                self.strategy_sample_counts[strategy] = len(self.current_indices)

            wandb.log({
                f"{strategy}_budget_{budget_percent:.2f}_epoch": epoch,
                f"{strategy}_budget_{budget_percent:.2f}_train_loss": train_loss,
                f"{strategy}_budget_{budget_percent:.2f}_train_acc": train_acc,
                f"{strategy}_budget_{budget_percent:.2f}_test_acc": test_acc,
                f"{strategy}_budget_{budget_percent:.2f}_f1": f1,
                f"{strategy}_budget_{budget_percent:.2f}_precision": precision,
                f"{strategy}_budget_{budget_percent:.2f}_recall": recall,
                f"{strategy}_budget_{budget_percent:.2f}_dataset_size": len(self.current_indices)
            })
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
        ps.print_stats(20)  # Print top 20 functions by cumulative time
        print(f"Profiling results for train_with_strategy ({strategy}):")
        print(s.getvalue())


    def should_add_samples(self, epoch: int, test_acc: float) -> bool:
        if len(self.current_indices) >= self.max_budget:
            return False
        if epoch < 10 or test_acc < 0.5:
            return epoch % 2 == 0  # Add samples every other epoch
        return (epoch + 1) % 10 == 0  # Less frequent additions in later stages

    def add_new_samples(self, new_samples: List[int]):
        self.current_indices = np.union1d(self.current_indices, new_samples).astype(int)
        self.update_train_loader()
        print(f"Added {len(new_samples)} samples. New dataset size: {len(self.current_indices)}")


    def run_comparison(self, budget_percents: List[float], epochs: int):
        for budget_percent in budget_percents:
            for strategy in self.all_strategies:
                self.train_with_strategy(strategy, budget_percent, epochs)
        
        self.plot_comparison_results(budget_percents)

    def plot_comparison_results(self, budget_percents: List[float]):
        plt.figure(figsize=(15, 10))
        for strategy in self.all_strategies:
            for budget in budget_percents:
                performances = self.strategy_performances[strategy]
                plt.plot(range(1, len(performances) + 1), performances, 
                         label=f"{strategy} (Budget: {budget:.2f})")
        
        plt.xlabel('Epochs')
        plt.ylabel('Test Accuracy')
        plt.title('Comparison of Selection Strategies and Budget Sizes')
        plt.legend()
        plt.grid(True)
        wandb.log({"comparison_plot": wandb.Image(plt)})
        plt.close()

        # Bar plot for final sample counts
        plt.figure(figsize=(12, 6))
        strategies = list(self.strategy_sample_counts.keys())
        counts = list(self.strategy_sample_counts.values())
        plt.bar(strategies, counts)
        plt.xlabel('Selection Strategy')
        plt.ylabel('Final Sample Count')
        plt.title('Final Sample Counts by Selection Strategy')
        wandb.log({"sample_count_plot": wandb.Image(plt)})
        plt.close()

# Usage
if __name__ == "__main__":
   wandb.init(project="interactive-curriculum-learning-competitors-meta")
   args = type('Args', (), {
        'lr': 0.001,
        'epochs': 100,
        'weight_decay': 1e-4,
        'dropout_rate': 0.5,
    })()

   trainer = InteractiveCIFAR10Trainer(args, api_key, budget_percent=0.3, initial_sample_size=1000)
   budget_percents = [0.1, 0.2, 0.3,0.4]
   trainer.run_comparison(budget_percents, args.epochs)