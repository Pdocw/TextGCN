dataset_name = 'prescription'
def load_data(file_path):
    labels = []
    texts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().split('\t')
            label = line[0]
            text = line[1]
            labels.append(label)
            texts.append(text)
    return labels, texts
labels, sentences = load_data('./prescription.txt')

train_or_test_list = ['train', 'test']


meta_data_list = []
n = len(labels)
for i in range(n):
    if i < 0.9 * n:
        meta = str(i) + '\t' + train_or_test_list[0] + '\t' + labels[i]
    else:
        meta = str(i) + '\t' + train_or_test_list[1] + '\t' + labels[i]
    meta_data_list.append(meta)

meta_data_str = '\n'.join(meta_data_list)

f = open('data/text_dataset/' + dataset_name + '.txt', 'w')
f.write(meta_data_str)
f.close()

corpus_str = '\n'.join(sentences)

f = open('data/text_dataset/corpus/' + dataset_name + '.txt', 'w')
f.write(corpus_str)
f.close()