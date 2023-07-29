from EmotionClassification.data_representation.data_representation import DataInstance
from nltk.tokenize import word_tokenize


class DemographicInformation:
    """
    This class provides methods to incorporate additional information in the naive bayes classifier
    for emotion classification

    As of the last edit the additional information is based on age, gender and education of the participant
    Furthermore, there is the option for proper tokenization based on the external nltk library

    The file was created on     Mon July 17th 2023
        it was last edited on   Thu July 27th 2023

    @author: Linnet M.
    """
    def __init__(self, filename, age=False, gender=False, education=False, tokenize=False):
        """
        this is the constructor for the class DemographicInformation which processes the file in the second step
        it calls the initial DataInstance class to access the data
        depending on the configuration, this class incorporates more information
            self.file_content_advanced stores a list containing every line in the file separated by tab
            self.extracted_data_advanced stores a list containing the data (a string)
                                         which is inserted in the NB calculation
            self.emotion_dependent_count_advanced stores two nested dictionaries counting tokens
                                                  depending on their emotion
                                                  used to calculate likelihood;
        it also contains three methods, stored in variables, for preprocessing the additional features

        :param filename: the filename to read the data from
        :param age: boolean variable to activate incorporation of age - default is False
        :param gender:  boolean variable to activate incorporation of gender - default is False
        :param education:  boolean variable to activate incorporation of education - default is False
        """
        self.age = age
        self.gender = gender
        self.education = education
        self.tokenize = tokenize

        self.data = DataInstance(filename, tokenize=self.tokenize)
        self.file_content_advanced = self.data.file_content

        self.preprocessed_age = self.bin_age()
        self.preprocessed_gender = self.categorize_gender()
        self.preprocessed_education = self.categorize_education()

        self.extracted_data_advanced = self.extract_data()
        self.emotion_dependent_count_advanced = self.emotion_dependent_frequency()

    def bin_age(self):
        """
        method to bin age

        for every instance in the column that contains age it replaces the integer by a category
        borders for the categories are chosen according to the equal frequency bins
        the word "Age" is put at the beginning to avoid ambiguity with other tokens in later steps

        :return prep_age: a dictionary that contains the preprocessed age category for each line {line:category}
        """

        prep_age = {}
        for line_num, line in enumerate(self.file_content_advanced, start=1):
            if line[45] != "age":
                age = int(line[45])
                if age <= 24:
                    prep_age[line_num] = "AgeYoung"
                elif 25 <= age <= 32:
                    prep_age[line_num] = "AgeMiddle"
                elif age >= 33:
                    prep_age[line_num] = "AgeOld"

        return prep_age

    def categorize_gender(self):
        """
        method to preprocess gender

        for every instance in the column that contains gender, the gender category is orthographically simplified or/and
        the feature label "Gender" is put before to avoid ambiguity with other tokens and cutting by the tokenizer in
        later steps

        :return prep_gender: dictionary that contains preprocessed gender category for each line {line:category}
        """
        prep_gender = {}
        for line_num, line in enumerate(self.file_content_advanced, start=1):
            if line[46] != "gender":
                gender = line[46]
                if gender == "Male":
                    prep_gender[line_num] = "GenderMale"
                elif gender == "Female":
                    prep_gender[line_num] = "GenderFemale"
                elif gender == "Gender Variant/Non-Conforming":
                    prep_gender[line_num] = "GenderQueer"
                elif gender == "Prefer not to answer":
                    prep_gender[line_num] = "GenderNone"

        return prep_gender

    def categorize_education(self):
        """
        method to preprocess education

        for every instance in the column that contains education, the education category is orthographically simplified
        or/and the feature label "Education" is put before to avoid ambiguity with other tokens and cutting by the
        tokenizer in later steps

        :return prep_education: dictionary that contains preprocessed education category for each line {line:category}
        """

        prep_education = {}
        for line_num, line in enumerate(self.file_content_advanced, start=1):
            if line[47] != "education":
                if line[47] == "High school":
                    prep_education[line_num] = "EducationHighSchool"
                if line[47] == "Undergraduate degree (BA/BSc/other)":
                    prep_education[line_num] = "EducationUndergraduateDegree"
                if line[47] == "Graduate degree (MA/MSc/MPhil/other)":
                    prep_education[line_num] = "EducationGraduateDegree"
                if line[47] == "Secondary education":
                    prep_education[line_num] = "EducationSecondaryEducation"
                if line[47] == "Doctorate degree (PhD/other)":
                    prep_education[line_num] = "EducationDoctorateDegree"
                if line[47] == "No formal qualifications":
                    prep_education[line_num] = "EducationNone"
                if line[47] == "Don't know / not applicable":
                    prep_education[line_num] = "EducationNone"

        return prep_education

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
            # count through the lines
            for line_num, line in enumerate(self.file_content_advanced, start=1):
                input_sequence = ""
                if line[1] == emotion and line[17] != "generated_text":
                    """
                    in the advanced step the goal is to incorporate more information such as 
                        age which is the age of the participant describing the event
                        gender which is the gender of the participant describing the event
                        education which is the education of the participant describing the event
                    """
                    if self.age and not self.gender and not self.education:
                        input_sequence += line[17]
                        # for every line add the corresponding preprocessed additional information
                        if line_num in self.preprocessed_age:
                            input_sequence += " " + self.preprocessed_age[line_num]
                    elif self.gender and not self.age and not self.education:
                        input_sequence += line[17]
                        if line_num in self.preprocessed_gender:
                            input_sequence += " " + self.preprocessed_gender[line_num]
                    elif self.education and not self.age and not self.gender:
                        input_sequence += line[17]
                        if line_num in self.preprocessed_education:
                            input_sequence += " " + self.preprocessed_education[line_num]
                    elif self.age and self.gender and not self.education:
                        input_sequence += line[17]
                        if line_num in self.preprocessed_age:
                            input_sequence += " " + self.preprocessed_age[line_num]
                        if line_num in self.preprocessed_gender:
                            input_sequence += " " + self.preprocessed_gender[line_num]
                    elif self.age and self.education and not self.gender:
                        input_sequence += line[17]
                        if line_num in self.preprocessed_age:
                            input_sequence += " " + self.preprocessed_age[line_num]
                        if line_num in self.preprocessed_education:
                            input_sequence += " " + self.preprocessed_education[line_num]
                    elif self.gender and self.education and not self.age:
                        input_sequence += line[17]
                        if line_num in self.preprocessed_gender:
                            input_sequence += " " + self.preprocessed_gender[line_num]
                        if line_num in self.preprocessed_education:
                            input_sequence += " " + self.preprocessed_education[line_num]
                    elif self.age and self.gender and self.education:
                        input_sequence += line[17]
                        if line_num in self.preprocessed_age:
                            input_sequence += " " + self.preprocessed_age[line_num]
                        if line_num in self.preprocessed_gender:
                            input_sequence += " " + self.preprocessed_gender[line_num]
                        if line_num in self.preprocessed_education:
                            input_sequence += " " + self.preprocessed_education[line_num]
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
        # count through the lines

        for line_num, line in enumerate(self.file_content_advanced, start=1):
            if line[17] != "generated_text":  # skip header
                input_sequence = ""
                if self.age and not self.gender and not self.education:
                    input_sequence += line[17].lower()
                    # for every line add the corresponding additional information
                    if line_num in self.preprocessed_age:
                        input_sequence += " " + self.preprocessed_age[line_num]
                elif self.gender and not self.age and not self.education:
                    input_sequence += line[17].lower()
                    if line_num in self.preprocessed_gender:
                        input_sequence += " " + self.preprocessed_gender[line_num]
                elif self.education and not self.age and not self.gender:
                    input_sequence += line[17].lower()
                    if line_num in self.preprocessed_education:
                        input_sequence += " " + self.preprocessed_education[line_num]
                elif self.age and self.gender and not self.education:
                    input_sequence += line[17].lower()
                    if line_num in self.preprocessed_age:
                        input_sequence += " " + self.preprocessed_age[line_num]
                    if line_num in self.preprocessed_gender:
                        input_sequence += " " + self.preprocessed_gender[line_num]
                elif self.age and self.education and not self.gender:
                    input_sequence += line[17].lower()
                    if line_num in self.preprocessed_age:
                        input_sequence += " " + self.preprocessed_age[line_num]
                    if line_num in self.preprocessed_education:
                        input_sequence += " " + self.preprocessed_education[line_num]
                elif self.gender and self.education and not self.age:
                    input_sequence += line[17].lower()
                    if line_num in self.preprocessed_gender:
                        input_sequence += " " + self.preprocessed_gender[line_num]
                    if line_num in self.preprocessed_education:
                        input_sequence += " " + self.preprocessed_education[line_num]
                elif self.age and self.gender and self.education:
                    input_sequence += line[17].lower()
                    if line_num in self.preprocessed_age:
                        input_sequence += " " + self.preprocessed_age[line_num]
                    if line_num in self.preprocessed_gender:
                        input_sequence += " " + self.preprocessed_gender[line_num]
                    if line_num in self.preprocessed_education:
                        input_sequence += " " + self.preprocessed_education[line_num]
                text_data.append(input_sequence)

        return text_data
