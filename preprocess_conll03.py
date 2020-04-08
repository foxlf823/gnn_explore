
import stanza
from tqdm import tqdm

def load_conll_data(file_path):
    instances = []
    words = []
    labels = []
    with open(file_path, 'r') as fp:
        for line in fp:
            line = line.strip()
            if len(line) == 0:
                if len(words) != 0:
                    instance = dict(words=words, labels=labels)
                    instances.append(instance)
                words = []
                labels = []
            else:
                columns = line.split()
                words.append(columns[0])
                labels.append(columns[-1])

    if len(words) != 0:
        instance = dict(words=words, labels=labels)
        instances.append(instance)

    nlp = stanza.Pipeline(lang='en', tokenize_pretokenized=True)

    for instance in tqdm(instances):
        doc = nlp([instance['words']])
        sentence = doc.sentences[0]
        dep_head = ['']*len(instance['words'])
        for i, word in enumerate(sentence.words):
            dep_head[i] = word.head - 1
        instance['heads'] = dep_head

    return instances

def dump_to_conll(instances, file_path):
    fp = open(file_path, 'w')
    for instance in instances:

        for word, head, label in zip(instance['words'], instance['heads'], instance['labels']):
            fp.write(word+" "+str(head)+" "+label+"\n")

        fp.write("\n")

    fp.close()

if __name__ == '__main__':
    # nlp = stanza.Pipeline('en')
    # doc = nlp("Barack Obama was born in Hawaii.")
    #
    # for sentence in doc.sentences:
    #     print(sentence.ents)
    #     print(sentence.dependencies)
    #     for word in sentence.words:
    #         print(word.text, word.lemma, word.pos)

    # nlp = stanza.Pipeline(lang='en', tokenize_pretokenized=True)
    # doc = nlp([['This', 'is', 'token.ization', 'done', 'my', 'way!'], ['Sentence', 'split,', 'too!']])
    # # doc = nlp(['This', 'is', 'token.ization', 'done', 'my', 'way!'])
    # for i, sentence in enumerate(doc.sentences):
    #     print(f'====== Sentence {i + 1} tokens =======')
    #     print(*[f'id: {token.id}\ttext: {token.text}' for token in sentence.tokens], sep='\n')

    # instances = load_conll_data("./conll03/debug.txt")
    # dump_to_conll(instances, './conll03/debug_my.txt')

    instances = load_conll_data("./conll03/train.txt")
    dump_to_conll(instances, './conll03/train_my.txt')

    instances = load_conll_data("./conll03/valid.txt")
    dump_to_conll(instances, './conll03/valid_my.txt')

    instances = load_conll_data("./conll03/test.txt")
    dump_to_conll(instances, './conll03/test_my.txt')