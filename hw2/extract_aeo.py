# Assignment 2  by Tianling Luo
# Main Logic
# In this assignment, I first set the important lists of words and parse the pdf document to get the texts.
# Then I deal with the raw text with some replacements modifications to make it more readable.
# Further, I pick the sentences that are related to specific energy type and the object (e.g. price, emission etc.)
# that we want to study.
# I use the functions to defect ancestors and children relationship with the key words,
# find if there are up/down/flat words, and then record the number.
# One noteworthy thing is that, I assign different weights to the ancestors, children,
# and even uncles (the children of ancestors' ancestor) relationships.
# For example, up/down/flat words in direct ancestors and children might be more important than
# those in uncles' relationships.
# After that, I get a result 2d-array recording the values we get from the analysis.
# While positive values mean "increase", negative values signify "decrease", and zero values and "flat".
# Then I transfer the 2d-array into a dataframe and then into a csv file.

# Weakness to improve:
# 1. This algorithm cannot record trends and fluctuations (e.g. first goes up for several years and then goes down)
# 2. It cannot judge if it is an assumption or a fact
# (e.g. "if the oil prices go up, ...; if the oil prices go down, ...")
# 3. The weight of ancestors, children, and uncles relationships for energy words and object words
# coming from empirical evidence and might be different in other contexts.

import PyPDF2
import pandas as pd
import spacy
import numpy as np
import os

PATH = r'D:\uchi\Fall\PPHA30536_Data and Programming for Public Policy II\homework-2-luotianling327'
UP = ['raise', 'increase', 'go up', 'higher', 'larger', 'greater', 'more', 'grow', 'growth', 'high']
DOWN = ['drop', 'decline', 'decrease', 'go down', 'lower', 'smaller', 'less', 'fall', 'low']
FLAT = ['unchanged', 'same', 'flat', 'stable']

def pdf_to_text(fname, n, m):
    pdf = PyPDF2.PdfFileReader(os.path.join(PATH, fname))
    text = []
    for pnum in range(n, m):
        page = pdf.getPage(pnum)
        text.append(page.extractText())
    full_text = ''.join(text)

    return full_text

def df_creation(n, column_list, energy_type):
    df = pd.DataFrame(index=range(n), columns=column_list)
    df['energy type'] = energy_type
    return df

def find_relevant_sents(sents, find_list):
    relevant_sents = {}
    for element in find_list:
        find_sents = []
        for sentence in sents:
            for token in sentence:
                if token.lemma_.lower() == element:
                    find_sents.append(sentence)
                    break
        relevant_sents[element] = find_sents
    return relevant_sents

def locate_word(word, sentence):
    index_list = []
    count = 0
    for token in sentence:
        if token.lemma_.lower() == word:
            index_list.append(count)
        count = count + 1
    return index_list

def initialize_matrix(m, n, value):
    result = np.empty([m, n])
    for i in range(m):
        for j in range(n):
            result[i][j] = value
    return result

def token_is_negated(token):
    if token.pos_ == 'VERB':
        for ancestor in token.ancestors:
            if ancestor.pos_ == 'VERB':
                for child in ancestor.children:
                    if child.dep_ == 'neg':
                        return True

    for child in token.children:
        if child.dep_ == 'neg':
            return True
    return False

def modify_trend(found, a_list, up_counter, down_counter, flat_counter, add):
    if add:
        for element in a_list:
            if element.lemma_.lower() in UP:
                found = True
                up_counter = up_counter + 1
            if element.lemma_.lower() in DOWN:
                found = True
                down_counter = down_counter + 1
            if element.lemma_.lower() in FLAT:
                found = True
                flat_counter = flat_counter + 1

    else:
        for element in a_list:
            if element.lemma_.lower() in UP:
                found = True
                up_counter = up_counter - 1
            if element.lemma_.lower() in DOWN:
                found = True
                down_counter = down_counter - 1
            if element.lemma_.lower() in FLAT:
                found = True
                flat_counter = flat_counter - 1
    return found, up_counter, down_counter, flat_counter

def analyze_trend(energy, relevant_sents):
    up_counter = 0
    down_counter = 0
    flat_counter = 0
    export_counter = 0
    import_counter = 0
    found = False

    for key in relevant_sents:
        if key == "export":
            for sent in relevant_sents[key]:
                index_export = locate_word(key, sent)
                if len(index_export) != 0:
                    found = True
                    for index in index_export:
                        token = sent[index]
                        if token_is_negated(token):
                            export_counter = export_counter - 1
                        else:
                            export_counter = export_counter + 1
            if not found:
                score = 'NaN'
            else:
                score = export_counter
            return score

        if key == "import":
            for sent in relevant_sents[key]:
                index_import = locate_word(key, sent)
                if len(index_import) != 0:
                    found = True
                    for index in index_import:
                        token = sent[index]
                        if token_is_negated(token):
                            import_counter = import_counter - 1
                        else:
                            import_counter = import_counter + 1
            if not found:
                score = 'NaN'
            else:
                score = import_counter
            return score

        else:
            for sent in relevant_sents[key]:
                index_list1 = locate_word(key, sent)
                index_list2 = locate_word(energy, sent)
                if energy == 'oil':
                    index_list2 = index_list2 + locate_word('petroleum', sent)
                ancestors_list1 = []
                ancestors_list2 = []
                children_list1 = []
                children_list2 = []
                uncles_list1 = []
                uncles_list2 = []
                for index1 in index_list1:
                    token1 = sent[index1]
                    ancestors_list1 = ancestors_list1 + list(token1.ancestors)
                    children_list1 = children_list1 + list(token1.children)
                    for ancestor1 in token1.ancestors:
                        for ancestor2 in ancestor1.ancestors:
                            uncles_list1 = uncles_list1 + list(ancestor2.children)
                for index2 in index_list2:
                    token2 = sent[index2]
                    ancestors_list2 = ancestors_list2 + list(token2.ancestors)
                    children_list2 = children_list2 + list(token2.children)
                    for ancestor1 in token2.ancestors:
                        for ancestor2 in ancestor1.ancestors:
                            uncles_list2 = uncles_list2 + list(ancestor2.children)

                combined_list = 7 * children_list1 + 5 * children_list2 + uncles_list1 + uncles_list2 + \
                                6 * ancestors_list1 + 4 * ancestors_list2

                if token_is_negated(token1):
                    found, up_counter, down_counter, flat_counter = modify_trend(found, combined_list, up_counter,
                                                                                 down_counter, flat_counter, False)

                else:
                    found, up_counter, down_counter, flat_counter = modify_trend(found, combined_list, up_counter,
                                                                                 down_counter, flat_counter, True)

    if not found:
        score = 'NaN'
    elif up_counter > down_counter and up_counter > flat_counter:
        score = up_counter - down_counter
    elif down_counter > up_counter and down_counter > flat_counter:
        score = up_counter - down_counter
    elif flat_counter > up_counter and flat_counter > down_counter:
        score = 0
    else:
        score = 0

    return score

def result_to_df(result, df, column_list):
    (m, n) = result.shape
    for i in range(m):
        for j in range(0, n-1):
            if j < n-2:
                if result[i][j] > 0:
                    df.loc[i, column_list[j+1]] = 'increase'
                elif result[i][j] < 0:
                    df.loc[i, column_list[j+1]] = 'decrease'
                elif result[i][j] == 0:
                    df.loc[i, column_list[j+1]] = 'flat'
            elif j == n-2:
                if str(float(result[i][j])).lower() == 'nan':
                    if result[i][j+1] > 0:
                        df.loc[i, column_list[j+1]] = 'import'
                elif str(float(result[i][j+1])).lower() == 'nan':
                    if result[i][j] > 0:
                        df.loc[i, column_list[j+1]] = 'export'
                elif result[i][j] - result[i][j+1] > 0:
                    df.loc[i, column_list[j+1]] = 'export'
                elif result[i][j] - result[i][j+1] < 0:
                    df.loc[i, column_list[j+1]] = 'import'
                elif result[i][j] - result[i][j+1] == 0:
                    df.loc[i, column_list[j+1]] = 'flat'
    return df

def df_to_csv(df, filename):
    df.to_csv(os.path.join(PATH, filename))


nlp = spacy.load("en_core_web_sm")
column_list = ["energy type", "price", "emissions", "production","export/import"]
energy_type = ["coal", "nuclear", "wind", "solar", "oil"]
key_words = {"price":["price", "cost", "costs", "prices"],
             "emissions":["emissions","emission","carbon","co2"],
             "production":["production","produce","generation"],
             "export":["export", "exporter", "net export"],
             "import":["import", "importer", "net import"]}

# For year 2019
text_2019 = pdf_to_text('aeo2019.pdf', 4, 14)
text_2019 = text_2019.replace('U.S. Energy Information Administration\nwww.eia.gov/aeo\n#AEO2019U.S. Energy Information Administration\n','')
text_2019 = text_2019.replace('\n','')
text_2019 = text_2019.replace('Ł','')
text_2019 = text_2019.replace('Š','')

doc_2019 = nlp(text_2019)
sents_2019 = list(doc_2019.sents)
energy_sents_2019 = find_relevant_sents(sents_2019, energy_type)
for sentence in sents_2019:
    for token in sentence:
        if token.lemma_.lower() == 'petroleum':
            energy_sents_2019['oil'].append(sentence)
            break
result_2019 = initialize_matrix(5, 5, 'NaN')

# Analyzing texts of the year 2019
count1 = 0
for energy in energy_type:
    count2 = 0
    r_sents_2019 = energy_sents_2019[energy]
    for key in key_words:
        k_words = key_words[key]
        relevant_sents_2019 = find_relevant_sents(r_sents_2019, k_words)
        score = analyze_trend(energy, relevant_sents_2019)
        result_2019[count1][count2] = score
        count2 = count2 + 1
    count1 = count1 + 1

df_2019 = df_creation(5, column_list, energy_type)
df_2019 = result_to_df(result_2019, df_2019, column_list)
df_to_csv(df_2019, 'result_2019.csv')

# For year 2018
text_2018 = pdf_to_text('aeo2018.pdf', 2, 16)
text_2018 = text_2018.replace('U.S. Energy Information Administration\nwww.eia.gov/aeo\n#AEO2018\nU.S. Energy Information Administration\n','')
text_2018 = text_2018.replace('\n','')
text_2018 = text_2018.replace('Ł','')
text_2018 = text_2018.replace('Š','')

doc_2018 = nlp(text_2018)
sents_2018 = list(doc_2018.sents)
energy_sents_2018 = find_relevant_sents(sents_2018, energy_type)
for sentence in sents_2018:
    for token in sentence:
        if token.lemma_.lower() == 'petroleum':
            energy_sents_2018['oil'].append(sentence)
            break
result_2018 = initialize_matrix(5, 5, 'NaN')

# Analyzing texts of the year 2018
count1 = 0
for energy in energy_type:
    count2 = 0
    r_sents_2018 = energy_sents_2018[energy]
    for key in key_words:
        k_words = key_words[key]
        relevant_sents_2018 = find_relevant_sents(r_sents_2018, k_words)
        score = analyze_trend(energy, relevant_sents_2018)
        result_2018[count1][count2] = score
        count2 = count2 + 1
    count1 = count1 + 1

df_2018 = df_creation(5, column_list, energy_type)
df_2018 = result_to_df(result_2018, df_2018, column_list)
df_to_csv(df_2018, 'result_2018.csv')