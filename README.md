# Text Summarization

The goal of this project is to  create text summary from article or document. It's written in Python and does not use any Machine Learning algorithms; only simple mathematical calculations (May be :-)); 

To make  text summary today you just have to find the right library (NLTK, Gensim, etc.) and with 4 lines of code the summary is done. This project will use a technique of text summarization for keeping the objective of simplicity and efficiency without library.

We will use an extractive method, i.e. the summary will be made from the sentences of the text and not by creating new sentences.

To select the relevant sentences in the text we use the [TF-IDF](https://fr.wikipedia.org/wiki/TF-IDF) method. This type of method consists in determining the relevance of a word in a document. Then we compare all the sentences with each other to determine the [cosine similarity](https://fr.wikipedia.org/wiki/Similarit%C3%A9_cosinus), in other term we compare of the sentences to finf there similarity.


Finally, we sort  sentences  ordered by  relevance.

**Note** : This project aim to work with French texts. However, it is easy to change the linguistic specificity.


## Data Processing

This project is divided into 5 step :

1. [Retrieve data and transform plain text into a phrase list](#Sentence retrieval)

2. [Filter data to keep important words](#Filtering data)

3. [Transform sentences into vectors](#Sentence vectorization)

4. [Create a similarity matrix between each sentence](#Similarity cosine)

5. [Sentence Sorting](#Sentence Sorting)

### Le texte

We have chosen a random newspaper article. The first sentence of the text is the title.

> Le constat d'échec de la justice dans la prévention des homicides conjugaux.
>
> Le rapport de l'inspection générale de la justice sur les homicides conjugaux sur 88 cas définitivement jugés pointe de graves dysfonctionnements dans la chaîne pénale.
>
> En décidant de rendre public, dimanche 17 novembre, le rapport de l'Inspection générale de la justice sur les homicides conjugaux, Nicole Belloubet, garde des sceaux, dévoile sans fard ce qui cloche dans la détection des signes annonciateurs de ces crimes. Et le constat est alarmant tant du côté des services de police et de gendarmerie, que du côté des magistrats, des services pénitentiaires ou même des services sociaux ou médicaux. "Très clairement, ça ne va pas. La chaîne pénale n’est pas satisfaisante", reconnaît la ministre de la justice dans un entretien publié le jour même dans Le Journal du dimanche.
>
> La mission de l'inspection a examiné 88 dossiers d'homicides conjugaux, de tentatives d'homicides et de violences volontaires ayant entraîné la mort sans intention de la donner commis en 2015 et 2016 et définitivement jugés depuis. 73 victimes étaient des femmes et 15, des hommes. Le traitement judiciaire du crime, une fois commis, semble plutôt satisfaisant.
>
> La durée moyenne de l'instruction judiciaire est de 17 mois (contre 31 mois pour les crimes en moyenne) et les sanctions prononcées par les cours d'assises traduisent une prise de conscience de la particularité de ces crimes. "La durée moyenne de réclusion criminelle est de 17 ans", lit-on dans le rapport, soit au-dessus de la moyenne des condamnations pour meurtres hors contexte conjugal. Surtout, les peines pour homicides conjugaux sont de plus en plus lourdes. Elles étaient de treize ans de réclusion en 2004.
>
> C'est en amont du meurtre que la justice n'est clairement pas à la hauteur. Car dans 63% des cas, des violences antérieures existaient, certes pas toujours dénoncées aux forces de l'ordre. Dans 35% des cas où des violences préexistaient, elles n’avaient pas été dénoncées à la police, mais étaient le plus souvent connues de la famille, des voisins ou de services sociaux. "Cette absence de dénonciation ou de signalement a empêché la mise en place de mesures susceptibles de prévenir l'homicide ultérieur", note l’inspection.
>
> L'absence de dénonciation par les médecins est également déplorée, alors qu'une dizaine de victimes de violences conjugales avaient auparavant consulté à l'hôpital ou en cabinet. Le rapport relate ainsi le cas d’une victime qui "s'est elle-même rendue à dix reprises aux urgences entre 2005 et 2014 dont quatre fois sur une année ; avant d’être tuée par arme à feu par son conjoint".


### Sentence retrieval

This first step consist in retrieving the article and breaking it down into sentences. A sentence should be considered as a series of signs ending either with a period (.), a question mark (?), an exclamation mark (!) or an end of line.

In order to split the sentences, it is necessary to consider exceptions:

 - Don't keep empty lines.
 - The sign "..." should not be considered as 3 sentence ends but only one sentence end.
 - Do not consider as an end of sentence a punctuation sign which is in a quote. For example: Son intervention était parsemée de " quand ? ", de  "pourquoi ? " et de " comment ? ". This exception is not managed in this project.
 - Sometimes a sentence may not be finished after a line break.


There are many other exceptions, which can be discovered by analyzing the datasets.

### Filtering data

The objective of this second step is to remove non relevant word in sentence.

For example for the sentence :

> C'est en amont du meurtre que la justice n'est clairement pas à la hauteur

 - Delete punctuation characters : ", ; :", etc.
 - Replace contracted forms with their extended form. Like  "n'" which becomes "ne".
 - Delete stop words (this list is provided by the NLTK library)
 - Replace all the words in lower case (the goal is to classify words so they must all be written the same way)
 - Special case of  "-" sign. Keep it or not depends opn word meaning. For example "vis-à-vis" is a complete word whereas "partirez-vous" sould be split in 2 parts. The "-" sign can also be used as a hyphenation. Theses cases are not managed in this project.

In this project the word "pas" has not been selected, it is a bias. However, it is generally necessary to keep the negatives form to avoid misunderstanding.

The sentence will become : "amont, meurtre, justice, clairement, hauteur"

Other techniques exist to fine tuning data filtering:

- [Lemmatisation](https://fr.wikipedia.org/wiki/Lemmatisation), will search the canonical form of the word. For example  "petites, petite, petits, petit" will be replaced by "petit". There is an online lemmatisation tool  (https://www.jerome-pasquelin.fr/tools/outil_lemmatisation.php)
- [Stemming](https://en.wikipedia.org/wiki/Stemming), will keep the root of the word. (ex. "clairement" will become "clair")

This is the most important part in data preparation and this is the part that takes the most time to analyze. As saying,  "Garbage in, garbage out." 

### Sentence vectorization
Once filtered sentences have been generated, the next step is to make sentence vector, i.e. to put each word of the sentence in a numerical form. There are many techniques to vectorize a sentence such as [Wod2vec](https://fr.wikipedia.org/wiki/Word2vec) or with neural networks like [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory). In this project we will use [TF-IDF](https://fr.wikipedia.org/wiki/TF-IDF). TF-IDF will calculate the pertinence of each word in the article and replace each word by this value.

TF-IDF processing is done as follows. We get the number of occurrences of a word in a sentence, then we will weight it with the occurrences of the same word in the other sentences. 

From previously filtered sentence: "amont, meurtre, justice, clairement, hauteur"

Result of TF-IDF processing:

| Word        | TF   | IDF  | TF-IDF |
| ---------- | ---- | ---- | ------ |
| amont      | 1    | 1    | 2.944  |
| meurtre    | 1    | 1    | 2.944  |
| justice    | 1    | 5    | 1,335  |
| clairement | 1    | 2    | 2.251  |
| hauteur    | 1    | 1    | 2.944  |

Most important words in this sentence are "upstream", "murder" and "height".  These three words are less used in the document, while we found 5 out of 19 sentences with the word  "justice"

The setence vector is: 

```
[ 2.944, 2.994, 1.335, 2.251,2.944]
```

### Similarity cosine
Once the sentences are vectorized, we will proceed to  [cosine similarity](https://fr.wikipedia.org/wiki/Similarit%C3%A9_cosinus). We will calculate the similarity of the sentences between them.

For the previously filtered sentence: "amont, meurtre, justice, clairement, hauteur which is in fact [ 2.944, 2.994, 1.335, 2.251,2.944 ].

The resulting matrix:

```
[0.91526243 0.41618927 0.44344526 0.55253342 0.80917498 0.64303634
 0.42371204 0.95539835 0.70505272 0.37176486 0.53289483 0.73716154
 0.97982853 0.         0.619055   0.52297626 0.50421168 0.5548457
 0.3616467 ]
```

More closer a number is to 1, stronger is the similarity. The element with a value of 0, corresponds to the sentence itself.

For our sentence: "C'est en amont du meurtre que la justice n'est clairement pas à la hauteur", the 3 most similar sentences are :

- 0,979 - Elles étaient de treize ans de réclusion en 2004
- 0,955 - 73 victimes étaient des femmes et 15, des hommes
- 0,915 - Le constat d'échec de la justice dans la prévention des homicides conjugaux

### Sentence Sorting
Finally, we will sum the cosine similarities of each sentence to obtain a ranking. Greater is the sum, more important the sentence is.

Our sentence: "C'est en amont du meurtre que la justice n'est clairement pas à la hauteur" is ranked at the 18th position (second last) . The cosine similarities sum is equal to 11.0481, so this sentence is clearly irrelevant.

## Text Summary
The 3 sentences text summary for  "**Le constat d'échec de la justice dans la prévention des homicides conjugaux**" looks like:

> L'absence de dénonciation par les médecins est également déplorée, alors qu'une dizaine de victimes de violences conjugales avaient auparavant consulté à l'hôpital ou en cabinet.
>
> "Cette absence de dénonciation ou de signalement a empêché la mise en place de mesures susceptibles de prévenir l'homicide ultérieur", note l’inspection.
>
> Dans 35% des cas où des violences préexistaient, elles n’avaient pas été dénoncées à la police, mais étaient le plus souvent connues de la famille, des voisins ou de services sociaux.


## Next Step
This project is a simple summary method that works in most cases, however, there is many improvement ways at each stage of construction. Adding more exceptions in data filtering is a good approach to get a "cleaner" text. Filtering is essential in data processing. In sentence vectorization other algorithms can be explored as [Wod2vec](https://fr.wikipedia.org/wiki/Word2vec), or [GloVe](https://nlp.stanford.edu/projects/glove/). Finally, in the sentence ranking, it is possible to use a "ranking" method.

In the end, the method depends on summarization type and on summarization precision 

In the end, the method to be used must depend on the type of text to be summarized and the degree of summary desired.