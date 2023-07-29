from EmotionClassification.main_work.naive_bayes import NaiveBayes
from EmotionClassification.data_representation.data_representation import DataInstance
from EmotionClassification.data_representation.emotion_info import EmotionInformation
from EmotionClassification.data_representation.demographic_info import DemographicInformation
import math


class Evaluation:
    """
    This class provides the evaluation of the naive bayes classifier implemented for emotion classification
    The evaluation is based on F1 scores per class as well as macro F1 scores

    The file was created on     Mon June  5th 2023
        it was last edited on   Sat July 29th 2023

    @author: Miriam S.
    """
    def __init__(self, filename_train, filename_test, event_duration=False, emotion_duration=False, intensity=False,
                 age=False, gender=False, education=False,
                 tokenize=False):
        """
        this is the constructor for the class Evaluation containing several important variables
        it calls the class NaiveBayes to access the NB calculation
                 the class DataInstance to access the general file contents (referring to both training and test file)
                 the class EmotionInformation to access additional information incorporated on demand
        :param filename_train: the name of the training file
        :param filename_test: the name of the test file
        :param event_duration: boolean variable to activate incorporation of event_duration - default is False
        :param emotion_duration:  boolean variable to activate incorporation of emotion_duration - default is False
        :param intensity:  boolean variable to activate incorporation of intensity - default is False
        :param age: boolean variable to activate incorporation of age - default is False
        :param gender:  boolean variable to activate incorporation of gender - default is False
        :param education:  boolean variable to activate incorporation of education - default is False
        :param tokenize: boolean variable to activate proper tokenization (using nltk tokenizer) - default is False
        """
        self.event_duration = event_duration
        self.emotion_duration = emotion_duration
        self.intensity = intensity
        self.age = age
        self.gender = gender
        self.education = education
        self.tokenize = tokenize

        # access the naive bayes calculation - all parameters' values are passed to the constructor
        self.naive_bayes_train = NaiveBayes(filename_train, event_duration=self.event_duration,
                                            emotion_duration=self.emotion_duration, intensity=self.intensity, age=self.age, gender=self.gender, education=self.education,
                                            tokenize=self.tokenize)
        # access the data in the baseline file with both training and test file for different purposes
        # data_train is used to reference all emotion labels that can be predicted (test file might not contain all)
        # data_test is used for reference of data to calculate the naive bayes for
        self.data_train = DataInstance(filename_train, tokenize=self.tokenize)
        self.data_test = DataInstance(filename_test, tokenize=self.tokenize)
        self.advanced_emo = EmotionInformation(filename_test, event_duration=self.event_duration,
                                               emotion_duration=self.emotion_duration, intensity=self.intensity,
                                               tokenize=self.tokenize)
        self.advanced_demo = DemographicInformation(filename_test, age=self.age, gender=self.gender,
                                                    education=self.education, tokenize=self.tokenize)
        # store all function outputs as variables
        self.predicted_labels = self.get_predicted()
        self.emotion_dict = self.calc_values_classes()
        self.precision = self.calc_precision()
        self.recall = self.calc_recall()
        self.f1 = self.calc_f1()

    def get_predicted(self):
        """
        get predicted labels for test instances
        :return: a list containing the predicted emotion labels for the instances in the test file
        """
        predicted_labels = []
        best_emotion, best_prob = None, -math.inf
        # get emotion labels from training file since this is what can be predicted
        emotion_labels = self.data_train.emotion_counts.keys()
        # check whether one of the additional information should be included and refer to corresponding file
        # advanced data contains additional data - non-advanced does not
        if self.event_duration or self.emotion_duration or self.intensity:
            data = self.advanced_emo.extracted_data_advanced
        elif self.age or self.gender or self.education:
            data = self.advanced_demo
        else:
            data = self.data_test.extracted_data

        # calculate the naive bayes probability for every sequence and every emotion to get the most likely emotion
        # iterate over every string (=sequence) in the total data from the test file as well as over every emotion
        for sequence in data:
            for emotion in emotion_labels:
                # get the naive bayes score from the corresponding function
                current_prob = self.naive_bayes_train.calculate_bayes(emotion, sequence)
                # get the best probability for emotion and store the label
                if best_prob < current_prob:
                    best_prob = current_prob
                    best_emotion = emotion
            # add the best emotion to a list and reset the values
            predicted_labels.append(best_emotion)
            best_emotion = None
            best_prob = -math.inf
        return predicted_labels

    def calc_f1(self):
        """
        calculate the f-score given the true and predicted labels
        :return: a dict containing emotions and corresponding f1 scores
        """
        emotion_f1 = dict()
        # iterate over possible emotions
        for emotion in set(self.data_test.true_emotion):
            p = self.precision[emotion]
            r = self.recall[emotion]
            # check whether precision and recall are 0 to avoid error of zero division and assign 0 immediately
            if p != 0 and r != 0:
                f1 = (2 * p * r) / (p + r)
            else:
                f1 = 0
            emotion_f1[emotion] = f1
        return emotion_f1

    def calc_values_classes(self):
        """
        helper method to calculate tp, fn, and fp for the respective classes separately
        :return: a dictionary containing the values corresponding to the emotion
        """
        tp, fn, fp = 0, 0, 0
        emotion_dict = dict()
        # iterate over possible emotions
        for emotion in set(self.data_test.true_emotion):
            for gold, pred in zip(self.data_test.true_emotion, self.predicted_labels):
                # increase the tp value if predicted and true emotion match
                if gold == pred and gold == emotion:
                    tp += 1
                # increase fn value if predicted and true don't match but true emotion is current emotion
                elif gold != pred and gold == emotion:
                    fn += 1
                # increase fp value if predicted and true don't match but predicted emotion is current emotion
                elif gold != pred and pred == emotion:
                    fp += 1
            emotion_dict[emotion] = {"tp": tp, "fp": fp, "fn": fn}
            # reset values for next emotion
            tp, fn, fp = 0, 0, 0
        return emotion_dict

    def calc_recall(self):
        """
        helper method to calculate recall for f1 score
        :return: a dictionary with emotions and corresponding recall scores
        """
        emotion_recall = dict()
        # iterate over possible emotions
        for emotion in set(self.data_test.true_emotion):
            tp = self.emotion_dict[emotion]["tp"]
            fn = self.emotion_dict[emotion]["fn"]
            # check whether tp and fn are 0 to avoid error of zero division and assign 0 immediately
            if tp != 0:
                recall = tp / (tp + fn)
            else:
                recall = 0
            emotion_recall[emotion] = recall
        return emotion_recall

    def calc_precision(self):
        """
        helper method to calculate precision for f1 score
        :return: a dictionary with emotions and corresponding precision scores
        """
        emotion_precision = dict()
        for emotion in set(self.data_test.true_emotion):
            tp = self.emotion_dict[emotion]["tp"]
            fp = self.emotion_dict[emotion]["fp"]
            # check whether tp and fp are 0 to avoid error of zero division and assign 0 immediately
            if tp != 0 and fp != 0:
                precision = tp / (tp + fp)
            else:
                precision = 0
            emotion_precision[emotion] = precision
        return emotion_precision

    def write_file(self, filename_output):
        """
        helper method to store output of evaluation in a separate .tsv file
        :param filename_output: the tsv-file's name for storing the output
        :return: returns a message after successfully writing file
        """
        gold = self.data_test.true_emotion
        file = open(filename_output, 'w')
        file.write("Emotion \t TP \t FP \t FN \t Precision \t Recall \t F1")
        for emotion in set(gold):
            fn = self.emotion_dict[emotion]["fn"]
            fp = self.emotion_dict[emotion]["fp"]
            tp = self.emotion_dict[emotion]["tp"]
            precision = "%.2f" % self.precision[emotion]
            recall = "%.2f" % self.recall[emotion]
            f1 = "%.2f" % self.f1[emotion]
            file.write("\n" + emotion + "\t" + str(tp) + "\t" + str(fp) + "\t" + str(fn) + "\t" + str(precision) + "\t"
                       + str(recall) + "\t" + str(f1))

        macro_precision = "%.2f" % (sum(self.precision.values()) / len(self.precision.values()))
        macro_recall = "%.2f" % (sum(self.recall.values()) / len(self.recall.values()))
        macro_f1 = "%.2f" % (sum(self.f1.values()) / len(self.f1.values()))
        file.write("\n\n" + "macro values" + "\t" + "\t" + "\t" + "\t" + str(macro_precision)
                   + "\t" + str(macro_recall) + "\t" + str(macro_f1))
        return "File {} was successfully written.".format(filename_output)



