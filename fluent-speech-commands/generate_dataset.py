import csv
from spacy.lang.en import English

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)


def extract_phrases(phrase_dict, csv_reader, word_list):
    """Extract phrases from CSV and tokenize files. Add duplicate phrases only once."""
    count_row = 0

    for row in csv_reader:
        phrase = row[3]
        count_row += 1

        if phrase not in all_phrases:
            tokens = tokenizer(phrase)
            tokens = list(tokens)
            phrase_dict[phrase] = tokens
            for tok in tokens:
                if tok not in words:
                    words.append(tok)

    #print(count_row)


test_data = './fluent_speech_commands_dataset/data/test_data.csv'
valid_data = './fluent_speech_commands_dataset/data/valid_data.csv'
train_data = './fluent_speech_commands_dataset/data/train_data.csv'
additional_phrases = 'additional_alterego_phrases.txt'
all_phrases = {}
words = []

with open(test_data) as csv_file:
    csv_reader_test = csv.reader(csv_file, delimiter=',')
    next(csv_reader_test, None)
    extract_phrases(all_phrases, csv_reader_test, words)

with open(valid_data) as csv_file:
    csv_reader_valid = csv.reader(csv_file, delimiter=',')
    next(csv_reader_valid, None)
    extract_phrases(all_phrases, csv_reader_valid, words)

with open(train_data) as csv_file:
    csv_reader_train = csv.reader(csv_file, delimiter=',')
    next(csv_reader_train, None)
    extract_phrases(all_phrases, csv_reader_train, words)

additional = open(additional_phrases, 'r').readlines()
for line in additional:
    line = line.strip()
    tokens = tokenizer(line)
    tokens = list(tokens)
    all_phrases[line] = tokens

print(len(all_phrases), 'phrases.')
print(len(words), 'unique tokens.')

output_file = open('fluent_speech_commands_wordmap.txt', 'w')
# Print word map for AlterEgo recording
for phrase, tokens in all_phrases.items():
    print('"' + phrase + ',' + ','.join(map(str, tokens)) + '", ', end="", file=output_file)

