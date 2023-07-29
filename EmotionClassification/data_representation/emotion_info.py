from EmotionClassification.data_representation.data_representation import DataInstance
from nltk.tokenize import word_tokenize


class EmotionInformation:
    """
    This class provides methods to incorporate additional information in the naive bayes classifier
    for emotion classification

    As of the last edit the additional information is based on event_duration, emotion_duration
    and intensity of the emotion
    Furthermore, there is the option for proper tokenization based on the external nltk library

    The file was created on     Sat July 1st 2023
        it was last edited on   Mon July 10th 2023

    @author: Miriam S.
    """
    def __init__(self, filename, event_duration=False, emotion_duration=False, intensity=False, tokenize=False):
        """
        this is the constructor for the class EmotionInformation which processes the file in the second step
        it calls the initial DataInstance class to access the data
        depending on the configuration, this class incorporates more information
            self.file_content_advanced stores a list containing every line in the file separated by tab
            self.extracted_data_advanced stores a list containing the data (a string)
                                         which is inserted in the NB calculation
            self.emotion_dependent_count_advanced stores two nested dictionaries counting tokens
                                                  depending on their emotion
                                                  used to calculate likelihood
        :param filename: the filename to read the data from
        :param event_duration: boolean variable to activate incorporation of event_duration - default is False
        :param emotion_duration:  boolean variable to activate incorporation of emotion_duration - default is False
        :param intensity:  boolean variable to activate incorporation of intensity - default is False
        """
        self.event_duration = event_duration
        self.emotion_duration = emotion_duration
        self.intensity = intensity
        self.tokenize = tokenize

        self.data = DataInstance(filename, tokenize=self.tokenize)
        self.file_content_advanced = self.data.file_content
        self.extracted_data_advanced = self.extract_data()
        self.emotion_dependent_count_advanced = self.emotion_dependent_frequency()

    def emotion_dependent_frequency(self):
        """
        helper method to count the frequency of tokens depending on the emotion
        :return: a dictionary of the form: {emotion: {token: count}}
        """
        total_dict, dependent_token_count = dict(), dict()
        emotions = set(self.data.true_emotion)
        # for every emotion create new dict with emotion as key
        # consequently add text to corresponding key
        for emotion in emotions:
            total_dict[emotion] = []
            for line in self.file_content_advanced:
                input_sequence = ""
                if line[1] == emotion and line[17] != "generated_text":
                    """
                    in the advanced step the goal is to incorporate more information such as 
                        event_duration which indicates the duration of the described event
                        emotion_duration which indicates the duration of the given emotion
                        intensity which indicates the intensity of the emotion felt/described
                    this information is added to the original text's string as an additional token
                    the information is prefixed with "event", "emotion", and "intensity" 
                    to make sure it is counted as the additional information
                    and not as one of the original tokens
                    """
                    if self.event_duration and not self.emotion_duration and not self.intensity:
                        input_sequence += line[17]
                        input_sequence += " event" + line[19]
                    elif self.emotion_duration and not self.event_duration and not self.intensity:
                        input_sequence += line[17]
                        # make sure value "I had none" is not split by tokenizer
                        if "none" in line[20]:
                            input_sequence += " emotionnone"
                        else:
                            input_sequence += " emotion" + line[20]
                    elif self.intensity and not self.emotion_duration and not self.event_duration:
                        input_sequence += line[17]
                        input_sequence += " intensity" + line[21]
                    elif self.event_duration and self.emotion_duration and not self.intensity:
                        input_sequence += line[17]
                        input_sequence += " event" + line[19]
                        # make sure value "I had none" is not split by tokenizer
                        if "none" in line[20]:
                            input_sequence += " emotionnone"
                        else:
                            input_sequence += " emotion" + line[20]
                    elif self.event_duration and self.intensity and not self.emotion_duration:
                        input_sequence += line[17]
                        input_sequence += " event" + line[19]
                        input_sequence += " intensity" + line[21]
                    elif self.emotion_duration and self.intensity and not self.event_duration:
                        input_sequence += line[17]
                        # make sure value "I had none" is not split by tokenizer
                        if "none" in line[20]:
                            input_sequence += " emotionnone"
                        else:
                            input_sequence += " emotion" + line[20]
                        input_sequence += " intensity" + line[21]
                    elif self.event_duration and self.intensity and self.emotion_duration:
                        input_sequence += line[17]
                        input_sequence += " event" + line[19]
                        # make sure value "I had none" is not split by tokenizer
                        if "none" in line[20]:
                            input_sequence += " emotionnone"
                        else:
                            input_sequence += " emotion" + line[20]
                        input_sequence += " intensity" + line[21]
                    total_dict[emotion].append(input_sequence)
        for emotion in total_dict.keys():
            # tokenize words properly if specified - split at whitespace otherwise
            if self.tokenize:
                total_dict[emotion] = word_tokenize(' '.join(total_dict[emotion]).lower())
            else:
                total_dict[emotion] = ' '.join(total_dict[emotion]).lower().split()
        # iterate over each emotion, collect emotion dependent vocabulary and create a new dict for each emotion
        for emotion in emotions:
            if emotion != "emotion":  # skip header
                words = total_dict[emotion]
                dependent_token_count[emotion] = dict()
                # iterate over every word - if the word is already part of the emotion dependent dictionary
                # increment the count, add a new entry otherwise
                for word in words:
                    if word in dependent_token_count[emotion].keys():
                        dependent_token_count[emotion][word] += 1
                    else:
                        dependent_token_count[emotion][word] = 1
        return dependent_token_count

    def extract_data(self):
        """
        helper method to extract the data for calculation
        :return: a list containing the data to calculate NB
        """
        text_data = []
        # extract the generated text to re-use for NB calculation
        for line in self.file_content_advanced:
            if line[17] != "generated_text":  # skip header
                input_sequence = ""
                if self.event_duration and not self.emotion_duration and not self.intensity:
                    input_sequence += line[17].lower()
                    input_sequence += " event" + line[19].lower()
                elif self.emotion_duration and not self.event_duration and not self.intensity:
                    input_sequence += line[17].lower()
                    # make sure value "I had none" is not split by tokenizer
                    if "none" in line[20].lower():
                        input_sequence += " emotionnone"
                    else:
                        input_sequence += " emotion" + line[20].lower()
                elif self.intensity and not self.emotion_duration and not self.event_duration:
                    input_sequence += line[17].lower()
                    input_sequence += " intensity" + line[21].lower()
                elif self.event_duration and self.emotion_duration and not self.intensity:
                    input_sequence += line[17].lower()
                    input_sequence += " event" + line[19].lower()
                    # make sure value "I had none" is not split by tokenizer
                    if "none" in line[20].lower():
                        input_sequence += " emotionnone"
                    else:
                        input_sequence += " emotion" + line[20].lower()
                elif self.event_duration and self.intensity and not self.emotion_duration:
                    input_sequence += line[17].lower()
                    input_sequence += " event" + line[19].lower()
                    input_sequence += " intensity" + line[21].lower()
                elif self.emotion_duration and self.intensity and not self.event_duration:
                    input_sequence += line[17].lower()
                    # make sure value "I had none" is not split by tokenizer
                    if "none" in line[20].lower():
                        input_sequence += " emotionnone"
                    else:
                        input_sequence += " emotion" + line[20].lower()
                    input_sequence += " intensity" + line[21].lower()
                elif self.event_duration and self.intensity and self.emotion_duration:
                    input_sequence += line[17].lower()
                    input_sequence += " event" + line[19].lower()
                    # make sure value "I had none" is not split by tokenizer
                    if "none" in line[20].lower():
                        input_sequence += " emotionnone"
                    else:
                        input_sequence += " emotion" + line[20].lower()
                    input_sequence += " intensity" + line[21].lower()
                text_data.append(input_sequence)

        return text_data

