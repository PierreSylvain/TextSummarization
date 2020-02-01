# Text Summarization

Ce projet consiste à créer des résumés de textes à partir d'articles ou de documents plus ou moins longs. Il est écrit en Python et n'utilise aucun algorithme de Machine Learning ; seulement des calculs mathématiques simples (enfin presque). 

Pour faire un résumé de texte aujourd'hui il suffit de prendre une bibliothèque adéquate (NLTK, Gensim, etc.) et avec 4 lignes de code le résumé est fait. Ce projet se propose d'étudier une des nombreuses technique de résumé de texte. L'objectifvest de rester simple et efficace.

Nous utilisons une méthode extractive, c'est à dire que le résumé sera fait à partir des phrases du texte et non pas en créant de nouvelles phrases.

Pour sélectionner les phrases pertinentes du texte nous utilisons un calcul  [TF-IDF](https://fr.wikipedia.org/wiki/TF-IDF). Ce tyep de calcul consiste à déterminer la pertinence d'un mot dans un document. Puis nous comparons toutes les phrases entre elles pour déterminer la [similarité de cosinus](https://fr.wikipedia.org/wiki/Similarit%C3%A9_cosinus). C'est à dire les phrases qui se ressemblent.

Enfin, nous faisons un tri pour afficher les phrases par ordre de pertinence. 

**Note** : Ce projet est prévu pour fonctionner avec des textes en français. Il est cependant facile de changer les spécificités linguistiques.


## Traitement des données

Pour réaliser ce projet nous l'avons divisé en 5 étapes :

1. [Récupération des données et transformation du texte brut en une liste de phrases](#Récupération des phrases)

2. [Filtrage des données pour ne conserver que les mots importants](#Filtrage des données)

3. [Transformation des phrases en vecteurs où chaque mot est remplacé par sa valeur de pertinence (TF-IDF)](#Vectorisation des phrases)

4. [Création d'une matrice de similarité entre chaque phrase du texte](#Similarité cosinus)

5. [Classement des phrases](Classement des phrases)

### Le texte

Ce texte est un article de journal pris au hasard. La première phrase du texte en est le titre.

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


### Récupération des phrases

Cette première étape consiste à récupérer l'article et a découper celui-ci en phrases. Il faut considérer une phrase comme une suite de signes terminés soit par le signe point (.), ou le signe point d'interrogation (?) ou le signe point d'exclamation (!) ou bien encore une fin de ligne.

Pour bien découper les phrases il est nécessaire de gérer des exceptions :
  - Ne pas prendre en compte les lignes vide.
  - Le signe "..." ne doit pas être considéré comme 3 fin de phrase mais une seule fin de phrase.
  - Ne pas considérer comme une fin de phrase une ponctuation qui serait dans une citation. Par exemple : Son intervention était parsemée de " quand ? ", de  "pourquoi ? " et de " comment ? "? Cette exception n'est pas gérée dans ce projet.
  - Prendre en compte le fait qu'une phrase n'est peut-être pas finie après un retour à la ligne.

Il existe de nombreuses autres exceptions, qui peuvent être découverte en analysant les jeux de données.

### Filtrage des données

Pour cette deuxième étape l'objectif est d'enlever des phrases les mots qui ne sont pas considérés comme pertinents. 

Nous prendrons comme exemple la phrase : 

> C'est en amont du meurtre que la justice n'est clairement pas à la hauteur

 - Supprimer les caractères de ponctuations : ", ; :", etc.
 - Remplacer les formes contractées par leur forme étendue. Comme par exemple "n'" qui devient "ne"
 - Supprimer les mots faisant partie des mots non pertinents (cette liste est fournie par la bibliothèque NLTK)
 - Remplacer tous les mots en minuscule (l'objectif est de classer des mots donc ils doivent tous être écrits de la même façon)
 -  Traiter le signe "-". C'est à dire le garder ou non suivant les cas. Par exemple "vis-à-vis" est un mot complet alors que "partirez-vous" pourrait être découper en 2 parties. Le trait d'union peut aussi être utilisé comme césure. Ce cas n'est pas traité dans ce projet.

Dans ce projet le mot "pas" n'a été conservé, c'est un parti pris. Cependant, il faut en général conserver les négations pour éviter les contres sens.

Après traitement la phrase devient : "amont, meurtre, justice, clairement, hauteur"

D'autres techniques existent pour affiner le filtrage des données :

- la [Lemmatisation](https://fr.wikipedia.org/wiki/Lemmatisation), qui consiste à chercher la forme canonique d'un mot. Par exemple les mots "petites, petite, petits, petit" sont remplacés par petit. Il existe un outil de lemmatisation en ligne (https://www.jerome-pasquelin.fr/tools/outil_lemmatisation.php)
- la [Stemmatisation](https://fr.wikipedia.org/wiki/Racinisation), qui consiste à ne garder que la racine des mots. Par exemple le mot : "clairement" devient "clair"

C'est la partie la plus importante dans la préparation des données et c'est cette partie qui prend le plus de temps à analyser. Car comme dit l'adage "Garbage in, garbage out" 

### Vectorisation des phrases
Une fois les phrases filtrées l'étape suivante consiste à faire de chaque phrase un vecteur, c'est à dire que l'on va mettre sous forme numérique chaque mot de la phrase. Il existe de nombreuses techniques pour vectoriser une phrase comme par exemple [Wod2vec](https://fr.wikipedia.org/wiki/Word2vec) ou en utilisant des réseaux de neurones de type [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory). Dans ce projet la technique utilisée est [TF-IDF](https://fr.wikipedia.org/wiki/TF-IDF). Le principe est de calculer le TF-IDF de chaque mot et de remplacer chaque mot par cette valeur.

Le calcul du TF-IDF est le suivant. On calcule le nombre d'occurences d'un mot dans une phrase, puis on va pondérer avec la fréquence d'apparition du mot dans les phrases. 

Pour la phrase filtrée précédemment :  "amont, meurtre, justice, clairement, hauteur"

Le calcul du TF-IDF donne :

| Mot        | TF   | IDF  | TF-IDF |
| ---------- | ---- | ---- | ------ |
| amont      | 1    | 1    | 2.944  |
| meurtre    | 1    | 1    | 2.944  |
| justice    | 1    | 5    | 1,335  |
| clairement | 1    | 2    | 2.251  |
| hauteur    | 1    | 1    | 2.944  |

Les mots les plus importants de cette phrase sont "amont", "meurtre" et "hauteur".  Ces trois mots sont les moins utilisés dans le document, alors que l'on retrouve le mot justice dans 5 phrases sur 19.

Le vecteur ainsi créé sera : 

```
[ 2.944, 2.994, 1.335, 2.251,2.944]
```


### Similarité cosinus
Comme les phrases sont vectorisées on va procéder au calcul de [similarité de cosinus](https://fr.wikipedia.org/wiki/Similarit%C3%A9_cosinus), cette à dire que l'on va calculer la similarité des phrases entre elles.

Pour la phrase filtrée précédement : "amont, meurtre, justice, clairement, hauteur" qui est en fait [ 2.944, 2.994, 1.335, 2.251,2.944 ]

On obtient la matrice suivante :

```
[0.91526243 0.41618927 0.44344526 0.55253342 0.80917498 0.64303634
 0.42371204 0.95539835 0.70505272 0.37176486 0.53289483 0.73716154
 0.97982853 0.         0.619055   0.52297626 0.50421168 0.5548457
 0.3616467 ]
```

Plus un chiffre est proche de 1, plus la similarité est forte. L'élément avec une valeur à 0, correspond à la phrase elle-même.

Pour la phrase de référence : "C'est en amont du meurtre que la justice n'est clairement pas à la hauteur", les 3 phrases les plus similaires sont :

- 0,979 - Elles étaient de treize ans de réclusion en 2004
- 0,955 - 73 victimes étaient des femmes et 15, des hommes
- 0,915 - Le constat d'échec de la justice dans la prévention des homicides conjugaux

### Classement des phrases
Enfin on calcule la somme des similarités cosinus de chaque phrase afin d'obtenir un classement. Plus la somme est importante plus la phrase est importante.

La phrase de référence :  "C'est en amont du meurtre que la justice n'est clairement pas à la hauteur' est en position 18 du classement (avant dernière) la somme des similarités cosinus égal à 11.0481, ce qui fait que clairement cette phrase n'est pas pertinente.

## Le résumé
Au final le résumé de 3 lignes pour "**Le constat d'échec de la justice dans la prévention des homicides conjugaux**" est le suivant :

> L'absence de dénonciation par les médecins est également déplorée, alors qu'une dizaine de victimes de violences conjugales avaient auparavant consulté à l'hôpital ou en cabinet.
>
> "Cette absence de dénonciation ou de signalement a empêché la mise en place de mesures susceptibles de prévenir l'homicide ultérieur", note l’inspection.
>
> Dans 35% des cas où des violences préexistaient, elles n’avaient pas été dénoncées à la police, mais étaient le plus souvent connues de la famille, des voisins ou de services sociaux.


## Étape suivante
Ce projet présente une méthode de résumé simple et qui fonctionne dans la plupart des cas, cependant  de nombreuses améliorations sont possibles à chaque étape de la construction. Ajouter plus d'exceptions dans le filtrage des données est une bonne approche pour obtenir un texte plus "propre". Le filtrage est l'élément essentiel du traitement des données. Dans la partie vectorisation des phrases d'autres algorithmes existent comme [Wod2vec](https://fr.wikipedia.org/wiki/Word2vec), ou [GloVe](https://nlp.stanford.edu/projects/glove/) par exemple. Enfin dans la partie classement des phrases, il est possible d'utiliser une méthode de "ranking".

Au final, la méthode à utiliser doit dépendre du type de texte à résumer et du degré de résumé que l'on souhaite.