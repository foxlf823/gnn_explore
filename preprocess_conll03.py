
import stanza


if __name__ == '__main__':
    nlp = stanza.Pipeline('en')
    doc = nlp("Barack Obama was born in Hawaii.")

    for sentence in doc.sentences:
        print(sentence.ents)
        print(sentence.dependencies)
        for word in sentence.words:
            print(word.text, word.lemma, word.pos)