from skseq.sequences.id_feature import IDFeatures
from skseq.sequences.id_feature import UnicodeFeatures
import re

# ----------
# Feature Class
# Extracts features from a labeled corpus (only supported features are extracted
# ----------
class ExtendedFeatures(IDFeatures):

    def add_emission_features(self, sequence, pos, y, features):
        x = sequence.x[pos]
        # Get tag name from ID.
        y_name = self.dataset.y_dict.get_label_name(y)

        # Get word name from ID.
        if isinstance(x, str):
            x_name = x
        else:
            x_name = self.dataset.x_dict.get_label_name(x)

        word = str(x_name)
        # Generate feature name.
        feat_name = "id:%s::%s" % (word, y_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)



        # Suffix demonyms
        demonym_suffix = ['an', 'ian', 'ine', 'ite', 'er', 'eno', 'ish', 'ese', 'i', 'ic', 'iote']
        check_suffix = list(filter(word.endswith, demonym_suffix)) != [] #check whether the word ends with some suffix
        if check_suffix == True:
            select_suffix = [x for x in demonym_suffix if word.endswith(x)]
            if len(select_suffix) == 2:
                select_suffix = ['ian']

            feat_name = "suffix:%s::%s" % (select_suffix[0], y_name)
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)


        #First letter is uppercase
        if word[0].isupper():
            feat_name = "capitalized:%s" % (y_name) #generate feature name
            feat_id = self.add_feature(feat_name) #get feature ID from name
            if feat_id != -1:
                features.append(feat_id)


        #All letters of the word are uppercase
        if word.isupper():
            feat_name = "uppercase:%s" % (y_name)
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)


        #The word has dots
        regex = re.compile('[\.]')
        check = regex.search(word) == None
        if check == False:
            feat_name = "Dot:%s" % (y_name)
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)


        #The word has hyphens
        regex = re.compile('[-]')
        check = regex.search(word) == None
        if check == False:
            feat_name = "Hyphen:%s" % (y_name)
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)


        #The string is all numeric
        if word.isnumeric():
            feat_name = "Numeric:%s::%s" % (len(word),y_name)
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)


        #The string has letters AND numbers
        letters = bool(re.search('[a-zA-Z]', word))
        numbers = any(char.isdigit() for char in word)
        if letters and numbers:
            feat_name = "LettersNumbers:%s" % (y_name)
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)

        #The word ends with -day:
        if word.endswith('day'):
            feat_name = "DaysWeek:%s" % (y_name)
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features.append(feat_id)




        return features







































class ExtendedUnicodeFeatures(UnicodeFeatures):

    def add_emission_features(self, sequence, pos, y, features):
        x = sequence.x[pos]
        # Get tag name from ID.
        y_name = y

        # Get word name from ID.
        x_name = x

        word = str(x_name)
        # Generate feature name.
        feat_name = "id:%s::%s" % (word, y_name)
        feat_name = str(feat_name)
        # Get feature ID from name.
        feat_id = self.add_feature(feat_name)
        # Append feature.
        if feat_id != -1:
            features.append(feat_id)

        if str.istitle(word):
            # Generate feature name.
            feat_name = "uppercased::%s" % y_name
            feat_name = str(feat_name)

            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        if str.isdigit(word):
            # Generate feature name.
            feat_name = "number::%s" % y_name
            feat_name = str(feat_name)

            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        if str.find(word, "-") != -1:
            # Generate feature name.
            feat_name = "hyphen::%s" % y_name
            feat_name = str(feat_name)

            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features.append(feat_id)

        # Suffixes
        max_suffix = 3
        for i in range(max_suffix):
            if len(word) > i+1:
                suffix = word[-(i+1):]
                # Generate feature name.
                feat_name = "suffix:%s::%s" % (suffix, y_name)
                feat_name = str(feat_name)

                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

        # Prefixes
        max_prefix = 3
        for i in range(max_prefix):
            if len(word) > i+1:
                prefix = word[:i+1]
                # Generate feature name.
                feat_name = "prefix:%s::%s" % (prefix, y_name)
                feat_name = str(feat_name)

                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features.append(feat_id)

        return features
