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
from openai import OpenAI
import time
import csv

class SelectionAgent:
    def __init__(self, strategy: str, api_key: str):
        self.strategy = strategy
        self.client = OpenAI(api_key=api_key)

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
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

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
        suggested_weights = eval(result.split('\n')[0])
        explanation = '\n'.join(result.split('\n')[1:])
        return suggested_weights, explanation

class MetaAgent:
    def __init__(self, strategies, learning_rate=0.1):
        self.strategies = strategies
        self.weights = {s: 1.0 / len(strategies) for s in strategies}
        self.learning_rate = learning_rate
        self.performance_history = {s: [] for s in strategies}

    def update_weights(self, rewards: Dict[str, float], suggested_weights: Dict[str, float]):
        for strategy in self.strategies:
            current_weight = self.weights[strategy]
            reward = rewards[strategy]
            suggested_weight = suggested_weights[strategy]
            
            # Combine the reward-based update with the moderator's suggestion
            new_weight = current_weight * (1 + self.learning_rate * reward)
            new_weight = 0.7 * new_weight + 0.3 * suggested_weight  # Blend current and suggested weights
            
            self.weights[strategy] = new_weight

        # Normalize weights
        total_weight = sum(self.weights.values())
        for strategy in self.strategies:
            self.weights[strategy] /= total_weight

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
        self.full_dataset = torchvision.datasets.CIFAR10(root='/Users/tanmoy/research/data', train=True, download=False, transform=transform_train)
        self.test_dataset = torchvision.datasets.CIFAR10(root='/Users/tanmoy/research/data', train=False, download=False, transform=transform_test)
        
        self.current_indices = self.smart_initial_sampling(self.args.initial_sample_size)
        self.update_train_loader()
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.workers)

    def setup_model(self):
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        self.model = self.model.to(self.device)

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)

    def setup_criterion(self):
        self.criterion = nn.CrossEntropyLoss()

    def setup_strategies(self):
        self.strategies = {
            "S_U": self.get_uncertain_samples,
            "S_C": self.get_class_balanced_samples,
            "S_B": self.get_diverse_samples,
            "S_D": self.get_misclassified_samples
        }

    def setup_agents(self):
        self.selection_agents = {
            "S_U": SelectionAgent("uncertainty", self.args.openai_api_key),
            "S_C": SelectionAgent("class balance", self.args.openai_api_key),
            "S_B": SelectionAgent("boundary", self.args.openai_api_key),
            "S_D": SelectionAgent("diversity", self.args.openai_api_key)
        }
        self.moderator = ModeratorAgent(self.args.openai_api_key)
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
        # Implement diversity sampling strategy (this is a placeholder)
        available_indices = list(set(range(len(self.full_dataset))) - set(self.current_indices))
        return np.random.choice(available_indices, min(n_samples, len(available_indices)), replace=False)

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

        # Calculate rewards (this is a placeholder - you might want to implement a more sophisticated reward system)
        rewards = {strategy: self.last_accuracy for strategy in self.strategies}

        # Update meta agent weights
        self.meta_agent.update_weights(rewards, suggested_weights)

        # Get final weights from meta agent
        final_weights = self.meta_agent.get_weights()

        # Log the proposals, moderator's decision, and final weights
        print("Agent Proposals:")
        for strategy, proposal in agent_proposals.items():
            print(f"{strategy}: {proposal}")
        print(f"Moderator Suggested Weights: {suggested_weights}")
        print(f"Moderator Explanation: {explanation}")
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
    parser.add_argument('--max-budget', type=int, default=5000, help='maximum number of samples to use (default: 25000)')
    parser.add_argument('--initial-sample-size', type=int, default=1000, help='initial number of samples (default: 1000)')
    parser.add_argument('--samples-per-epoch', type=int, default=1000, help='number of samples to add per epoch (default: 1000)')
    parser.add_argument('--save-path', type=str, default='./checkpoints', help='path to save checkpoints')
    parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint (default: none)')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency (default: 10)')
    parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
    parser.add_argument('--openai-api-key', type=str, required=True, help='OpenAI API key for agent communication')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    trainer = CurriculumCIFAR10Trainer(args)
    trainer.run()

if __name__ == '__main__':
    main()