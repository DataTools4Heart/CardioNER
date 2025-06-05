import os
import re
import spacy
import pandas as pd


# nlp = spacy.load('ro_core_news_lg') # Romanian
# nlp = spacy.load('xx_sent_ud_sm') # Multilingual -> Czech
# nlp = spacy.load('sv_core_news_lg') # Swedish
# nlp = spacy.load('it_core_news_lg') # Italian
# nlp = spacy.load('es_core_news_lg') # Spanish
# nlp = spacy.load('en_core_web_lg') # English
nlp = spacy.load('nl_core_news_lg') # Dutch


# attribute_list = ['MEDICATION']
# attribute_list = ['DISEASE']
attribute_list = ['PROCEDURE']

# path_list = [r'../data/CardioCCC_Dataset/Training/Dutch/txt']
path_list = [r'../data/CardioCCC_Dataset/Testing/Dutch/txt']

# annotations_path = "../data/CardioCCC_Dataset/Training/Dutch/cardioccc_nl_proc.tsv"
annotations_path = "../data/CardioCCC_Dataset/Testing/Dutch/cardioccc_nl_proc.tsv"
annotations_df = pd.read_csv(annotations_path, sep='\t')


def main():
    # output_file = '../data/CardioCCC_Dataset/Training/Dutch/' + 'cardioccc_nl_proc_bio_format.tsv'
    output_file = '../data/CardioCCC_Dataset/Testing/Dutch/' + 'cardioccc_nl_proc_bio_format.tsv'
    with open(output_file, 'w', encoding='utf8') as output:
        for path in path_list:
            for _, report_text_file in enumerate(os.listdir(path)):
                with open(os.path.join(path, report_text_file), "r", encoding='utf8') as myfile:
                    report_text = myfile.readlines()
                    report_text = ''.join(report_text)
                    report_text = report_text.replace('vs.', 'vs ')
                full_stop_positions = [x.start() for x in re.finditer('\.', report_text)]
                full_stop_positions = [0] + full_stop_positions + [len(report_text)]
                if len(full_stop_positions) > 0:
                    for position_index in range(1, len(full_stop_positions)):
                        sentence_start = full_stop_positions[position_index - 1]
                        sentence_end = full_stop_positions[position_index]
                        text = report_text[sentence_start:sentence_end + 1]
                        if len(text) > 5:
                            # get annotations
                            # annotations = annotations_df[annotations_df['name']==report_text_file.replace('.txt','')] # for training files
                            annotations = annotations_df[annotations_df['filename']==report_text_file.replace('.txt','')] # for test files
                            idx = 0
                            entities = {}
                            for index, line in annotations.iterrows():
                                # annotation_type = line['tag'] # for training files
                                annotation_type = line['label'] # for test files
                                annotation_start = int(line['start_span'])
                                annotation_end = int(line['end_span'])
                                if annotation_start >= sentence_start and annotation_end <= sentence_end:
                                    entities[idx] = {'type': annotation_type, 'start': annotation_start - sentence_start, 'end': annotation_end - sentence_start}
                                    idx = idx + 1

                            doc = nlp(text)
                            token_list = []
                            for token in doc:
                                if token.text=='.':
                                    if token.idx!=0:
                                        token_list.append((token.text, token.idx, token.idx + len(token)))
                                elif "\n" not in token.text:
                                    token_list.append((token.text, token.idx, token.idx + len(token)))
                            output_tokens = ['O'] * len(token_list)
                            for entity in entities.values():
                                start = int(entity['start'])
                                end = int(entity['end'])
                                switch = 0
                                for index, token in enumerate(token_list):
                                    if int(token[1]) >= start and int(token[2]) <= end and entity['type'] in attribute_list and switch == 0:
                                        output_tokens[index] = 'B-' + entity['type']
                                        switch = 1
                                        continue
                                    if int(token[1]) >= start and int(token[2]) <= end and entity['type'] in attribute_list and switch == 1:
                                        output_tokens[index] = 'I-' + entity['type']
                            for token, label in zip(token_list, output_tokens):
                                output.write(token[0] + ' _ _ ' + label + '\n')
                            output.write('\n')

if __name__=="__main__":
    main()