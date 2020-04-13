from pathlib import Path
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib import colors
import itertools


class AbductiveReasoner(object):
    """
    Abductive Reasoner. Uses neighbouring cell states for generating
    explanations to predict the output for this competition
    https://www.kaggle.com/c/abstraction-and-reasoning-challenge
    created by it-from-bit - https://www.kaggle.com/everyitfrombit.

    Attributes
    ----------
    observations (dict): stores observations as keys and conclusions as values.
    explanations (dict): using observations, creates explanations and stores as keys
    along with the conclusions as values.

    """
    def __init__(self, kernel, config):
        """
        Abductive Reasoner.

        Attributes
        ----------
        data_path (str): Path to the dataset.
        kernel (class): Kernel instance for nearest neighbours (cell features).
        """
        self.cmap = colors.ListedColormap(
            ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
             '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
        self.norm = colors.Normalize(vmin=0, vmax=9)
        self.task = None
        self.observations = None
        self.explanations = None
        self.color_explanations = None
        self.input_original = None
        self.input_ = None
        self.output = None
        self.config = config
        self.kernel = kernel
        self.data_path = Path(self.config["dataset"])
        self._load_tasks()

    def _load_tasks(self):
        train_path, valid_path = self.data_path / "training", self.data_path / "evaluation"
        test_path = self.data_path / "test"
        self.train_tasks = {task.stem: json.load(task.open()) for task in train_path.iterdir()}
        self.valid_tasks = {task.stem: json.load(task.open()) for task in valid_path.iterdir()}
        if test_path.exists():
            self.test_tasks = {task.stem: json.load(task.open()) for task in test_path.iterdir()}

    def init_task(self, file, is_test=False):
        self.task = self.train_tasks.get(file) if file in self.train_tasks else self.valid_tasks.get(file)
        if is_test:
            self.task = self.test_tasks[file]
        self.observations, self.explanations = None, None
        self.input_, self.output = None, None

    def _pad_image(self, image):
        return np.pad(image, self.kernel.d, constant_values=0)

    def _sample_handler(self, sample):
        self.input_ = np.array(sample["input"])
        self.input_ = self._pad_image(self.input_)
        if "output" in sample:
            self.output = np.array(sample["output"])
            self.output = self._pad_image(self.output)
        else:
            self.output = None
        self.input_original = self.input_.copy()
        if self.kernel.memory:
            self.kernel.memory.init_memory(self.input_original)

    def _grid_walk(self, direction):
        rows, cols = self.input_.shape[0], self.input_.shape[1]
        r0 = reversed(range(self.kernel.d, rows-self.kernel.d)) if direction[:2] == "bt" else range(self.kernel.d, rows-self.kernel.d)
        for i in r0:
            r1 = reversed(range(self.kernel.d, cols - self.kernel.d)) if direction[2:] == "rl" else range(self.kernel.d,cols - self.kernel.d)
            for j in r1:
                yield i, j

    def _generate_observation(self, neighs, conclusion=None):
        return neighs.tolist() + [conclusion] if conclusion is not None else neighs.tolist()

    def observe(self):
        num_loops = self.config.get("num_observation_loops", 10)
        walk_directions = self.config.get("observation_walk_directions", ["tblr"])
        train = self.task["train"].copy()
        self.observations = []
        for d in walk_directions:
            for sample in train:
                self._sample_handler(sample)
                for loop in range(num_loops):
                    for i, j in self._grid_walk(d):
                        neighs = self.kernel.get_neighbours(self.input_, i, j)
                        conclusion = self.kernel.get_label(self.output, i, j)
                        if self._sum_neighs(neighs) > 0:
                            observation = self._generate_observation(neighs, conclusion)
                            if observation not in self.observations:
                                self.observations.append(observation)
                            self.input_[i, j] = self.output[i, j]
        self.input_ = self.input_original.copy()  # reset input

    def create_observation_df(self, is_sorted=False):
        df = pd.DataFrame(
            np.array(self.observations),
            columns=[f"feature_{i}" for i in range(len(self.observations[0][:-1]))] + ["conclusion"])
        df = df.sort_values(list(df.columns)) if is_sorted else df
        return df

    def _combination_walk(self):
        r_min = self.config.get("combination_r_min", 1)
        r_max = self.config.get("combination_r_max", self.kernel.k)
        for r in range(r_min, r_max+1):
            for combi in itertools.combinations(self.kernel.indices, r):
                yield combi

    def _explanation_handler(self, explanations, freq_threshold):
        explanations_ = explanations.copy()
        for explanation in explanations.keys():
            if len(set(explanations[explanation]["conclusion"])) == 1:  # no contradiction condition
                freq = len(explanations_[explanation]["conclusion"])
                if freq >= freq_threshold:
                    explanations_[explanation]["frequency"] = freq
                    explanations_[explanation]["conclusion"] = int(explanations[explanation]["conclusion"][0])
                else:
                    del explanations_[explanation]
            else:
                del explanations_[explanation]
        return explanations_

    def _generate_explanation(self, observation, combi):
        explanation = ",".join(
            [str(s) if i in combi else "-" for i, s in enumerate(observation[:self.kernel.k])])
        return explanation

    def reason(self):
        freq_threshold = self.config.get("frequency_threshold", 2)
        explanations = {}
        for combi in self._combination_walk():
            for observation in self.observations:
                explanation = self._generate_explanation(observation, combi)
                if explanation in explanations:
                    explanations[explanation]["conclusion"].append(observation[-1])
                else:
                    explanations[explanation] = {"conclusion": [observation[-1]]}
        self.explanations = self._explanation_handler(explanations, freq_threshold)

    def create_explanation_df(self, is_sorted=False, is_color=False):
        explanations = self.color_explanations if is_color else self.explanations
        rows = []
        for explanation in explanations.keys():
            con = explanations[explanation]["conclusion"]
            freq = explanations[explanation]["frequency"]
            rows.append([s for s in explanation.split(",")] + [con] + [freq])
        columns = [f"feature_{i}" for i in range(self.kernel.k)] + ["conclusion"] + ["frequency"]
        df = pd.DataFrame(np.array(rows), columns=columns)
        df = df.sort_values(list(df.columns)) if is_sorted else df
        return df

    def _observation_encoder(self, observation):
        d = list(dict.fromkeys(observation))
        encoded_observation = [str(d.index(c)) for c in observation]
        return encoded_observation, d

    def explain_color(self):
        freq_threshold = self.config.get("color_frequency_threshold", 2)
        explanations = {}
        for combi in self._combination_walk():
            for observation in self.observations:
                observation, _ = self._observation_encoder(observation)
                explanation = self._generate_explanation(observation, combi)
                if explanation in explanations:
                    explanations[explanation]["conclusion"].append(observation[-1])
                else:
                    explanations[explanation] = {"conclusion": [observation[-1]]}
        self.color_explanations = self._explanation_handler(explanations, freq_threshold)

    def _decide_conclusion(self, conclusions):
        conclusion = None
        val = - np.inf
        df = pd.DataFrame(conclusions, columns=["conclusion", "frequency", "level"])
        for conc in df.conclusion.unique():
            val_ = df[(df.conclusion == conc) & (df.level == "color_level")].frequency.shape[0]
            if not val_:
                val_ = df[(df.conclusion == conc) & (df.level == "first_level")].frequency.shape[0]
            if val_ > val:
                conclusion, val = conc, val_
        return conclusion

    def _sum_neighs(self, neighs):
        len_memory = self.kernel.len_memory
        return neighs[len_memory if len_memory != 0 else None:].sum()

    def _remove_padding(self, frame):
        return frame[self.kernel.d: -self.kernel.d, self.kernel.d: -self.kernel.d]

    def _revert_sample_padding(self):
        self.input_original = self._remove_padding(self.input_original)
        self.input_ = self._remove_padding(self.input_)
        if self.output is not None:
            self.output = self._remove_padding(self.output)

    def _compute_score(self, prediction):
        score = None
        if self.output is not None:
            self._revert_sample_padding()
            score = 1 if np.array_equal(self.output, prediction) else 0
        return score

    def predict(self, is_train=False, visualize=False):
        num_loops = self.config.get("num_inference_loops", 10)
        visualize_prediction = self.config.get("visualize_prediction", False)
        walk_directions = self.config.get("prediction_walk_directions", ["tblr"])
        samples = self.task["test"] if not is_train else self.task["train"]
        predictions, scores = [], []
        for d in walk_directions:
            for sample in samples:
                self._sample_handler(sample)
                prediction = self.input_.copy()
                for loop in range(num_loops):
                    for i, j in self._grid_walk(d):
                        neighs = self.kernel.get_neighbours(prediction, i, j)
                        if self._sum_neighs(neighs) > 0:
                            explanation_set, conclusions = [], []
                            for combi in self._combination_walk():
                                observation = self._generate_observation(neighs)
                                explanation = self._generate_explanation(observation, combi)
                                encoded_obs, color_set = self._observation_encoder(observation)
                                encoded_explanation = self._generate_explanation(encoded_obs, combi)
                                try:
                                    con = self.color_explanations[encoded_explanation]["conclusion"]
                                    freq = self.color_explanations[encoded_explanation]["frequency"]
                                    con = color_set[con]
                                    conclusions.append((con, freq, "color_level"))
                                    explanation_set.append(encoded_explanation)
                                except:
                                    if explanation in self.explanations:
                                        con = self.explanations[explanation]["conclusion"]
                                        freq = self.explanations[explanation]["frequency"]
                                        conclusions.append((con, freq, "first_level"))
                                        explanation_set.append(explanation)
                            conclusion = self._decide_conclusion(conclusions)
                            prediction[i, j] = conclusion if conclusion is not None else prediction[i, j]
                            if visualize_prediction:
                                self.plot_sample(prediction)
                prediction = self._remove_padding(prediction)
                predictions.append(prediction)
                scores.append(self._compute_score(prediction))
                if visualize:
                    self.plot_sample(prediction)
        return predictions, scores

    def save_prediction(self, prediction):
        self.plot_sample(prediction)

    def plot_pictures(self, pictures, labels):
        fig, axs = plt.subplots(1, len(pictures), figsize=(2 * len(pictures), 32))
        for i, (pict, label) in enumerate(zip(pictures, labels)):
            axs[i].imshow(np.array(pict), cmap=self.cmap, norm=self.norm)
            axs[i].set_title(label)
        plt.show()

    def save_pictures(self, pictures, labels, path):
        fig, axs = plt.subplots(1, len(pictures), figsize=(2 * len(pictures), 2 * len(pictures)))
        for i, (pict, label) in enumerate(zip(pictures, labels)):
            axs[i].imshow(np.array(pict), cmap=self.cmap, norm=self.norm)
            axs[i].set_title(label)
        fig.savefig(path)
        plt.close()

    def plot_sample(self, predict=None):
        pictures = [self.input_original, self.output] if self.output is not None else [self.input_original]
        labels = ['Input', 'Output'] if self.output is not None else ["Input"]
        if predict is not None:
            pictures = pictures + [predict]
            labels = labels + ["Predict"]
        self.plot_pictures(pictures, labels)

    def save_samples(self, path, predictions=None, is_train=False):
        samples = self.task["test"] if not is_train else self.task["train"]
        predictions = [None] * len(samples) if predictions is None else predictions
        for i, (prediction, sample) in enumerate(zip(predictions, samples)):
            self._sample_handler(sample)
            self._revert_sample_padding()
            pictures = [self.input_original, self.output] if self.output is not None else [self.input_original]
            labels = ['Input', 'Output'] if self.output is not None else ["Input"]
            if prediction is not None:
                pictures = pictures + [prediction]
                labels = labels + ["Predict"]
            path_ = Path(path).parent / f"{Path(path).stem}_{i}{Path(path).suffix}"
            self.save_pictures(pictures, labels, path_)

    def plot_train(self):
        train = self.task["train"]
        for sample in train:
            self._sample_handler(sample)
            self.plot_sample()
