from nltk.tokenize import word_tokenize


def read_file(filename):
    complete_data = []
    # read data and store it in a list (line by line)
    with open(filename, encoding="utf8") as f:
        for lines in f:
            line = lines.split('\t')
            complete_data.append(line)
    return complete_data


class DataInstance:
    """
        This class provides methods to read-in and process the data from the input file
        The data is read in for the baseline model and does not come with additional features

        The file was created on     Mon June  5th 2023
            it was last edited on   Mon July 10th 2023

        @author: Miriam S.
        """
    def __init__(self, filename, tokenize=False):
        """
        this is the constructor for the class DataInstance which reads in the file in the first step
        it stores instance variables containing data content from the different files
            self.file_content stores a list containing every line in the file separated by tab
            self.extracted_data stores a list containing the data (a string) which is inserted in the NB calculation
            self.true_emotions stores a list containing all emotions labels serving as true emotions in the evaluation
            self.emotion_counts stores a dictionary with emotions together with their respective count in the file
                                used to calculate prior probability in naive bayes
            self.emotion_dependent_count stores two nested dictionaries counting tokens depending on their emotion
                                         used to calculate likelihood
        :param filename: the file to read in the data from
                         can be both training and test file depending on the specification in other files
        """
        self.tokenize = tokenize
        self.file_content = read_file(filename)
        self.extracted_data = self.extract_data()
        self.true_emotion = self.extract_emotion()
        self.emotion_counts = self.emotion_frequency()
        self.emotion_dependent_count = self.emotion_dependent_frequency()

    def extract_data(self):
        """
        helper method to extract the data for calculation
        :return: a list containing the data to calculate NB
        """
        text_data = []
        # extract the generated text to re-use for NB calculation
        for line in self.file_content:
            if line[17] != "generated_text":  # skip header
                text_data.append(line[17].lower())
        return text_data

    def extract_emotion(self):
        """
        helper methods to extract the emotions in the correct order from the inserted file
        :return: a list of extracted emotions
        """
        emotions = []
        # extract the generated text to re-use for NB calculation
        for line in self.file_content:
            if line[1] != "emotion":  # skip header
                emotions.append(line[1])
        return emotions

    def emotion_frequency(self):
        """
        count individual emotions
        :return: a dictionary of the form {emotion: count}
        """
        emotion_count = dict()
        # count the emotions
        # by adding one to the count if the emotion was found previously
        # or creating a new dictionary entry otherwise
        for token in self.file_content:
            if token[1] != "emotion":  # skip header
                if token[1] not in emotion_count.keys():
                    emotion_count[token[1]] = 1
                else:
                    emotion_count[token[1]] += 1
        return emotion_count

    def emotion_dependent_frequency(self):
        """
        helper method to count the frequency of tokens depending on the emotion
        :return: a dictionary of the form: {emotion: {token: count}}
        """
        total_dict, dependent_token_count = dict(), dict()
        emotions = set(self.true_emotion)
        # for every emotion create new dict with emotion as key
        # consequently add text to corresponding key
        for emotion in emotions:
            total_dict[emotion] = []
            for line in self.file_content:
                if line[1] == emotion and line[17] != "generated_text":
                    total_dict[emotion].append(line[17])
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

