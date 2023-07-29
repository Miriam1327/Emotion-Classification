from EmotionClassification.data_representation.data_representation import DataInstance
from EmotionClassification.data_representation.emotion_info import EmotionInformation
from EmotionClassification.data_representation.demographic_info import DemographicInformation
import math
from nltk.tokenize import word_tokenize


class NaiveBayes:
    """
        This class is used to calculate the naive bayes scores for the emotion classification

        The file was created on     Mon June  5th 2023
            it was last edited on   Sat July 29th 2023

        @author: Miriam S.
        """
    def __init__(self, filename, event_duration=False, emotion_duration=False, intensity=False, age=False, gender=False,
                 education=False, tokenize=False):
        """
        this is the constructor for the class Evaluation containing several important variables
        it calls the class DataInstance to access the general file contents (referring to both training and test file)
                 the class EmotionInformation to access additional emotion-dependent information incorporated on demand
                 the class DemographicInformation to access additional demographic information incorporated on demand
        :param filename: the name of the file
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

        self.data_class = DataInstance(filename, tokenize=self.tokenize)
        self.advanced_emo = EmotionInformation(filename, event_duration=self.event_duration,
                                               emotion_duration=self.emotion_duration, intensity=self.intensity,
                                               tokenize=self.tokenize)
        self.advanced_demo = DemographicInformation(filename, age=self.age, gender=self.gender,
                                                    education=self.education, tokenize=self.tokenize)

    def calculate_prior(self, emotion):
        """
        helper method to calculate the prior probability needed for Naive Bayes calculation
        prior probability P(A) is calculated by dividing the count of the class label by the number of all labels
        :param emotion: the label to calculate the prior probability for NB (one of the emotions)
        :return: the prior probability (frequency of the given label)
        """
        emotion_frequencies = self.data_class.emotion_counts
        emotion_freq = emotion_frequencies[emotion]
        total_count = sum(emotion_frequencies.values())
        return math.log(emotion_freq) - math.log(total_count)

    def calculate_likelihood(self, emotion, data):
        """
        helper method to calculate the likelihood (conditional) needed for Naive Bayes calculation
        likelihood P(A|B) is calculated by dividing the count of the emotion given the data divided by
         the count for the data P(emotion|data) = P(data, emotion) / P(data)
        :param emotion: the given emotion to calculate the likelihood for NB (one of the emotions)
        :param data: the given data (a sentence dependent on the emotion)
        :return: the likelihood for the corresponding label and data
        """
        # P(data, emotion) is calculated such that the log probabilities
        # of the individual words given the emotion are added
        prob_data_emotion = 0
        # access the dictionary with emotion dependent token counts
        if self.event_duration or self.emotion_duration or self.intensity:
            emotion_dependent_token_count = self.advanced_emo.emotion_dependent_count_advanced
        elif self.age or self.gender or self.education:
            emotion_dependent_token_count = self.advanced_demo.emotion_dependent_count_advanced
        else:
            emotion_dependent_token_count = self.data_class.emotion_dependent_count
        # get the dependent counts for tokens given the emotion
        total_dependent_token_count = sum(emotion_dependent_token_count[emotion].values())
        # get the vocabulary count given the emotion
        dependent_vocabulary_count = len(emotion_dependent_token_count[emotion].keys())
        # tokenize words properly if specified - split at whitespace otherwise
        if self.tokenize:
            data = word_tokenize(data.lower())
        else:
            data = data.lower().split()
        # add each log probability to get the final probability
        for word in data:
            if word in emotion_dependent_token_count[emotion].keys():
                prob_data_emotion += math.log((emotion_dependent_token_count[emotion][word] + 1) /
                                              (total_dependent_token_count + dependent_vocabulary_count))
            else:
                prob_data_emotion += math.log(1 / (total_dependent_token_count + dependent_vocabulary_count))
        return prob_data_emotion

    def calculate_bayes(self, emotion, data):
        """
        calculate the conditional probability of the given emotion occurring given the data
        :param emotion: the corresponding emotion
        :param data: the data (a sequence to calculate the naive bayes for)
        :return: the conditional log probability
        """
        likelihood = self.calculate_likelihood(emotion, data)
        prior = self.calculate_prior(emotion)

        return likelihood + prior

