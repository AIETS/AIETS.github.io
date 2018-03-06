---
layout: post
title: Introduction à l'apprentissage machine
img_header: ![Nicolas Cage en Lois Lane dans Man Of Steel]({{ site.baseurl }}/images/article1/nicolas_cage_in_superman.png)
desc: Comment Nicolas Cage peut-il se retrouver là?!?
---

## Ce qu'on va voir!
1. [Définition](#def)
2. [Comment ça marche, à peu près?](#comment-ca-marche)
3. [La suite](#suite)

## 1. <a name="def"></a>Définition

Si vous êtes ici, j'imagine qu'il n'est pas nécessaire de vous vanter les avantages de l'apprentissage machine, vous les connaissez sans doute déjà. Par contre, comme lorsqu'on apprend à programmer pour la première fois, il est probable que ce soit difficile d'imaginer comment tout cela fonctionne.

Dans mon cours de communication, on m'a dit que de commencer une présentation avec une citation n'était pas une bonne façon de faire. Toutefois, je ne suis pas vraiment un bon élève :

> Apprentissage machine : Champ d'études qui donne aux ordinateurs l'habilité d'apprendre sans être explicitement programmés. (Samuel, 1959, traduit de l'anglais)[1](http://ieeexplore.ieee.org/document/5392560/)

Deux points ont retenu mon attention la première fois que j'ai vu cette citation :

1. 1959?! Pourtant, je n'avais jamais entendu ce terme avant 2016 (j'étais probablement en retard sur la mode, il faut dire). La réalité, c'est que les techniques d'apprentissage machine existent depuis bien longtemps, mais les ordinateurs n'étaient pas assez puissants pour permettre aux algorithmes d'être performants.
2. "Sans être explicitement programmés". Pourtant, il doit bien y avoir de la programmation quelque part, non? Évidemment, la réponse est oui! Toutefois, c'est probablement assez loin de ce que vous vous imaginez.

<span style="color:blue">*(Parenthèse)*</span>

**J'espère (ou ne pas) vous surprendre en vous informant du fait que vous avez très probablement déjà implémenté un algorithme d'apprentissage machine par vous-même, avant même que vous ayez eu vos connaissances en informatique.**

*Throwback* aux cours de maths en 5e secondaire, introduction aux statistiques. Si vous avez fait votre secondaire au Québec dans les... au moins 15 dernières années, vous avez sans doute été introduit aux deux concepts suivants : *la corrélation linéaire* et la *régression linéaire*.

Et bien, la régression linéaire est une technique d'apprentissage machine, vous êtes (presque) déjà des experts. [Yoshua Bengio](https://mila.quebec/personne/bengio-yoshua/) et [Yann LeCun](http://yann.lecun.com) n'ont qu'à bien se tenir! (Si vous ne les connaissez pas, il s'agit de personnalités par mal connue dans le milieu. En plus, monsieur Bengio est une fierté de chez nous, du québec, pas de l'ÉTS)

Dans le prochain article de ce blogue, nous reviendrons sur les détails et l'implémentation de la régression linéaire. (**EXCITEMENT**)

<span style="color:blue">*(Fin de la parenthèse)*</span>

L'apprentissage machine, c'est un sujet profond *(pun intended)*. Peut-être que je veux que mon algorithme puisse :

1. [Conduire une voiture dans GTA5](https://github.com/gtarobotics/self-driving-car)
2. [Prédire le prix du bitcoin](https://github.com/cbyn/bitpredict)
3. [Te proposer le nouveau film (plate) d'Adam Sandler sur Netflix](https://www.rtinsights.com/netflix-recommendations-machine-learning-algorithms/)
4. [Remplacer les faces d'acteurs par Nicholas Cage dans des films](http://www.indiewire.com/2018/01/nicolas-cage-machine-learning-algorithm-deep-fakes-1201923224/)

Ces quatre exemples n'utilisent vraisemblablement pas les mêmes techniques (focus sur les 3 premiers, le dernier est là parce que ça me fait rire) et n'ont pas les mêmes objectifs.

![Nicolas Cage en Lois Lane dans Man Of Steel]({{ site.baseurl }}/images/article1/nicolas_cage_in_superman.png)

---

## 2. <a name="comment-ca-marche"></a>Comment ça marche, à peu près?

*Préparez-vous pour le parallèle le plus incroyable au monde*

Imaginez un bébé. Lorsqu'on est enfant, on apprend à parler en écoutant nos parents parler (disclaimer : je ne suis pas un pro de bébés). Plus nos parents répètent des mots, plus il y a de chances que le bébé finisse par les répéter.

**Le bébé apprend par observation!**

Deuxième exemple : Prenons le même bébé fictif, qui apprend maintenant à marcher. C'est pas mal certain que la première fois que le bébé va marcher, il va tomber. En tombant, il se fait mal. On peut donc dire qu'il est puni parce qu'il est tombé. À vouloir éviter les punitions, le bébé va s'adapter et finir par être capable de marcher. Il apprend donc à marcher à force de se tromper et de se corriger (comme [ce bot](https://backyardrobotics.eu/2017/11/27/build-a-balancing-bot-with-openai-gym-pt-i-setting-up/))

**Le bébé apprend par renforcement!**

![Bébé qui apprend à marcher]({{ site.baseurl }}/images/article1/baby-walking.jpg )


Maintenant, imaginons un monde fictif où il n'y a pas de règles sur la maltraitance des enfants. Si je prenais ce même bébé, mais que je l'enfermais dans une pièce en lui présentant des millions de photos de chiens. Pour ces millions de photos, on lui indiquerait toujours "Ceci est un [insérer la race du chien]". À un point (encore une fois, je ne suis pas un pro de bébés), on pourrait montrer des nouvelles photos de chiens au bébé et il serait capable de dire de quel type de chien il s'agit, mais il ne serait pas capable de faire quoi que ce soit d'autre :'(.

Ce dernier exemple représente parfaitement l'apprentissage machine. L'ordinateur apprend avec des exemples identifiés, jusqu'au point où il est capable de faire des prédictions lui-même.

Comment ça se passe concrètement? On y reviendra dans l'article sur la régression linéaire, ça mérite un long texte en soi.

---

## 3. <a name="suite"></a>La suite

Cet article fut de courte durée! J'ai réfléchi "longuement" à savoir si je devrais rajouter du contenu pour la rendre plus charnue. Je ne pense pas que ce soit une bonne idée. Le but de ces articles est de fournir une approche simple vers l'apprentissage machine et je crois que ce sera plus facile de faire cela si les articles sont de courtes durées avec de l'information concise.

Dans le prochain article, nous reviendrons sur la régression linéaire. On y abordera aussi les techniques d'optimisation en apprentissage machine. Bref, ça risque d'être beaucoup plus intéressant!

---

Francis
