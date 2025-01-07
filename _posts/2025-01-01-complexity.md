---
layout: post
title: "Complexité"
subtitle: "introduction à la notion de complexité d'algorithme."
date: 2020-01-26 23:45:13 -0400
background: '/img/posts/complexity/complexity.png'
---

# Introduction sur complexité algorithmique

### Qu’est-ce qu’une complexité algorithmique ?

C’est un outil ou modèle mathématique important en science informatique qui permet d’estimer les performances asymptotiques d’un algorithme. Par asymptotique, on parle des grandes valeurs de l’instance d’entrée qui tendent vers l’infini. Ce modèle permettrait alors de comparer les algorithmes qui résolvent un même problème.

En d’autres mots, c’est un outil qui va nous permettre d’estimer le temps que prendra l’algorithme à exécuter toutes les instructions, on parlera de la complexité temporelle. D’un autre côté, cet outil va permettre d’estimer l’espace mémoire nécessaire que prendra l’algorithme pour exécuter toutes les instructions, on parlera de la complexité spatiale.

En parlant d’instruction ou d'opération élémentaire, pour illustrer ces concepts, prenons quelques exemples d’algorithmes, dont la multiplication des entiers et la multiplication des matrices. Les opérations élémentaires à considérer (affectation, incrémentation, comparaison, etc.) ne seront pas les mêmes. Dans le premier cas, on pourrait prendre la multiplication des bits (suite de 0 et 1) et dans le second cas la multiplication des entiers en base 10. Pour aller plus loin, prenons un algorithme de NLP, dans ce cas, l’opération élémentaire pourrait être la multiplication des vecteurs représentant les mots. Pour un dernier exemple, l’algorithme de tri, l’opération élémentaire serait le nombre de comparaisons.

> Petit exemple parlant, prenons le cas des LLM (Large Language Model), leurs complexités temporelles et spatiales ne leur permettent pas de fonctionner correctement sur un pc classique, puisqu'on manquera d’espace mémoire nécessaire pour charger tout le modèle en mémoire, et même si on arrive à charger le modèle, même pour les petits modèles, la génération du texte prendra plusieurs minutes, ce qui les rendrait impraticables. (Connaissez-vous les complexités du modèle GPT-3.5 ?) Les opérations élémentaires considérées ?)
> 

<aside>
💡 Par abus de langage, lorsqu'on parle de la complexité tout court dans la littérature, les blogs, etc., on fait référence à la complexité temporelle.

</aside>

Donc, par complexité, on entend une fonction qui va compter le nombre des opérations élémentaires de notre algorithme lorsque l’instance d’entrée est très grande.

Il existe trois modèles mathématiques de complexité : 

1.   $\Omega$  est utilisé afin d’exprimer la complexité dans le meilleur des cas. Intuitivement, ce modèle va nous dire ceci : pour n’importe quelle instance du problème, nous ne pourrons pas faire mieux (moins d’opérations) que la borne définie. Il définit la borne inférieure asymptotique de la complexité de notre algorithme.
2. $O$  est utilisé pour exprimer la complexité dans le pire des cas. Contrairement à  $\Omega$, ce modèle va nous dire ceci : pour n’importe quelle instance du problème, nous ne pourrons pas faire pire que la borne définie. Il définit la borne supérieure asymptotique de la complexité de notre algorithme.
3.  $\Theta$ est utilisé pour exprimer la complexité dans le pire et le meilleur cas. Pour n’importe quelle instance, le modèle indiquera que la complexité de  l’algorithme est majorée et minorée par une même fonction g à deux constantes multiplicatives. Dans le meilleur des cas, ta complexité ne sera pas meilleure qu’une fonction g et dans le pire des cas, elle ne sera pas pire que la fonction g à deux constantes multiplicatives. Ta complexité est prise en sandwich par la fonction g à deux constantes multiplicatives. 

Ce qui est le plus intéressant, c’est la complexité dans le pire des cas, puisque si nous avons une bonne complexité dans le pire des cas, l’algorithme ne peut qu’être meilleur dans le meilleur des cas ou en moyenne (certains algorithmes sont meilleurs que d’autres que dans le meilleur des cas). 

<aside>
💡 **Par abus de langage, complexité=complexité temporelle dans le pire des cas**

</aside>

> En tant que Machine Learning Engineer (MLE) ou Data Scientist (DS), il est important de penser à la complexité de l’algorithme qu’on met en place, qu’il s’agisse d’une architecture d’un modèle, ou d’un algorithme d’entrainement ou d’inférence.
> 

### **Formulation mathématique de la compléxité**

Soient $g$ et $f$ deux fonction de $\N$ dans $\R$, 

**Complexité dans le meilleur cas** 

$$
f \in \Omega(g) = \exists c>0,\ \exists N \in \N ,\ \forall n \ge N ,\ 0\le c g(n) \le f(n)    
$$

$f \in \Omega(g)$ ou $f = \Omega(g)$,   si et seulement si à partir d’un certain rang la fonction $f$ croît plus rapidement que la fonction $g$ à une constante multiplicative près $c$. La fonction $cg(n)$  définit une borne inférieure de la fonction $f$ à partir du rang N. 

Dans le cadre de l’algorithmie, $\Omega$  va nous informer que pour des instances de plus en plus grandes pour un algorithme telle que sa complexité est exprimée par la fonction $f$ , $f$ ne pourra pas croitre plus vite que la fonction  $g$. En pratique, ce modèle, cette notation est utilisée pour obtenir une borne inférieure dans le meilleur des cas. Il s’agit aussi d’une borne inférieure pour n’importe quel autre cas.

**Complexité dans le pire cas** 

$$
f \in O(g) = \exists c>0,\ \exists N \in \N ,\ \forall n \ge N ,\ 0\le f(n) \le c g(n)
$$

$f \in O(g)$ ou $f = O(g)$, si et seulement si à partir d’un certain rang la fonction $g$ croît plus rapidement que la fonction $f$ à une constante multiplicative près $c$. La fonction $cg(n)$  définit une borne supérieure de la fonction $f$ à partir du rang N. 

Dans le cadre de l’algorithmie, $\Omega$  va nous informer que pour des instances de plus en plus grandes, pour un algorithme dont sa complexité est exprimée par la fonction $f$ , $f$ ne pourra pas croitre plus vite que la fonction  $g$. En pratique, ce modèle, cette notation est utilisée pour obtenir une borne supérieure dans le pire des cas. Il s’agit aussi d’une borne supérieure pour n’importe quel autre cas.

**Complexité dans le pire et meilleur cas** 

$$
f \in \Theta(g) = \exists c,d>0,\  \exists N \in \N ,\ \forall n \ge N ,\ d g(n)\le f(n) \le c g(n)
$$

$f \in \Theta(g)$ ou $f = \Theta(g)$, si et seulement si à partir d’un certain rang les deux fonctions $f$ et $g$ ont la même croissance asymptotiquement, multipliée par deux constantes. La fonction $f$ est prise en sandwich par la fonction $g$ à deux constantes multiplicatives près. En d’autres mots, $f$ est majorée et minorée par $g$ à deux constantes multiplicatives près, à partir du rang N.

En algorithmique, $\Theta$ permet d’exprimer une complexité dans le pire et le meilleur des cas. Pour n’importe quelle instance du problème, dans tous les cas, la complexité est minorée et majorée par $g$ à deux constantes multiplicatives près.

### Exemple théorique

Soient deux fonctions $f=x^3$ et $g=\dfrac{1}{200}x^4$ positives définies sur $\R$. 

![Untitled](/img/posts/complexity/f_g_functions.png)

A partir du $n=200$, la fonction $g$ croît plus rapidement que la fonction $f$, et inversement.  On peut affirmer que $f$ est en $O(g)$ et $g$ est en $\Omega(f)$ pour $N=200$, et $c=1$. 

### Exemple d’un cas pratique d’analyse d’algorithme

1. **Rechercher d’un élément dans un tableau**

Nous avons un tableau contenant les éléments triés et nous souhaitons savoir si l’élément est présent dans le tableau. En entrée, un tableau trié et un élément à rechercher dans ce tableau. Il s’agit d’un problème classique, rechercher un élément dans une base de données par exemple.

**1.1 Analyse d’algorithme version 1**

```python
def recherche_v1(L:List[int],e:int):	
    
    for elt in L:
        if elt==e:
            return True
    return False
```

Comptons le nombre d’opérations élémentaires de cet algorithme. On notera $f$ ****la fonction qui détermine le nombre d’opérations de notre algorithme.

```markup
A chaque tour de boucle
    1 affectation ( la variable elt )
    1 comparaison
Et une seule instruction return.
```

Le meilleur des cas est celui où la valeur recherchée se trouve à la première position du tableau. On aura un seul tour de boucle, donc 2 opérations en tout.  $f= 3$. Donc, peu importe la taille de la liste, notre fonction n’exécutera que 3 instructions. Elle est en $\Theta(1)$ dans le meilleur des cas, donc $O(1)$  et en $\Omega(1)$. Il suffit de choisir $g=c$, où $c$ est une constante.

Le pire des cas est celui où la valeur recherchée n’est pas présente dans la liste. Donc, on aura $n$ tour de boucle pour un tableau de taille $n$,  2  opérations qui s’exécutent n fois et une opération de return:  $2n + 1$.  Donc $f=2n+1$. La fonction f dépend donc de la taille du tableau, elle dépend donc de n. On écrira  $f(n)=2n+1$.  Elle est en $\Theta(n)$ dans le meilleur des cas, donc $O(n)$  et en $\Omega(n)$. Pour prouver que $f$ est $\Theta(n)$, on pourrait choisir $g(n)=n$, $d=2$, $c=3$, $N=2$. 

Peu importe l’entrée, la complexité est en $O(n)$ et en $\Omega(1)$. Dans la pratique, on s’intéresse au cas le plus défavorable, on cherche donc la borne supérieure de notre algorithme, on notera $O(n)$ tout simplement comme étant la complexité de l’algorithme. Notez qu’on ne pourra pas dire que cet algorithme est pas en $\Theta(n)$ (Pourquoi ? réponse laissée au lecteur).

**1.2 Algorithme version 2 ( version récursive )**

```python
def recherche_v2(L:List[int],e:int):
    
    res = len(L)
    if res==1:
        return L[0]==e
    
    if L[res//2]==e:
        return True
    
    if L[res//2]>e:
        return recherche_v2(L[:res//2],e)
    else:
        return recherche_v2(L[res//2:],e)
```

```markup
A chaque appel de fonction
    1 affectation ( la variable elt )
    3 comparaison
    une instruction return.
```

Le meilleur des cas est celui où l’élément recherché se trouve à la première position du tableau, donc on n’aura qu’un seul appel de la fonction, et deux opérations élémentaires. On a une complexité constante  $\Theta(1)$ dans le meilleur des cas.

Dans le pire des cas, on a 4 opérations à chaque appel. Combien y a-t-il d’appels de fonction ? À chaque appel, la taille est divisée par 2. Si $n = [4,5,6,7]$ on a 2 appels, si $n = [8,...,15]$ on a trois appels, etc. Donc pour une liste de taille $n$ on a $\lfloor log (n) \rfloor$ appels.

Pour un tableau de taille $n$ on a $4*\lfloor log (n) \rfloor$  opérations élémentaires. On notera $f(n) = 4*\lfloor log (n) \rfloor$, la complexité de notre algorithme est de  $\Theta(logn)$.

Avec la même analyse faite pour l’algorithme pour la version 1, avec la complexité dans le meilleur des cas $\Theta(1)$  et dans le pire des cas $\Theta(logn)$, on peut conclure que la complexité de cet algorithme version 2 est  $O(logn)$.

1. **$x^n \mod y$**

On souhaite calculer  **x puissance n modulo y.** Ce type de calcul est très utilisé dans la cryptographie, la théorie des nombres, etc. On va proposer deux versions d’algorithmes et ensuite évaluer leurs complexités

**2.1 Analyse version 1**

```python
def puissance_v1(x:int,n:int):
    
    if n==0: return 1
    
    res = x
    for i in range(n-1):
        res=res*x
        res=res%27
    return res
```

```markup
1 comparaison
1 instruction return
A chaque tour de boucle
    3 affectation ( les variable i et res)
    1 multiplication
    1 divsion euclidienne (modulo) 
une instruction return.
```

La boucle est exécutée n fois, donc on a $3 + 5n$ instructions au total. Dans le meilleur comme dans le pire des cas, l’algorithme exécutera n tours de boucle pour une valeur n, donc la complexité est en $O(n)$.

**2.2 Analyse version 2**

```python
def puissance_v2(x:int,n:int):
    
    if n==0: return 0
    if n==1: return 1
    
    tmp = puissance_v2(x,n//2)
    if n%2==0: 
        return tmps*tmps%27
    else:
        return tmps*tmps*x%27
```

```markup
A chaque appel de fonction
    2 comparaisons
    (1 ou 2 instructions return, pour n=1,2)
    1 affectation ( la variable tmp)
    1 test
    2 divisions euclidiennes
    2 ou 3 multiplication
    1 instruction return.
```

À chaque appel de fonction, nous avons 9 instructions élémentaires, à 1 ou 2 près. Combien d’appels de fonction sont faits ? Il y a $\lfloor log (n) \rfloor$ appels, puisqu’à chaque appel $n$ est divisé par deux pour l’appel suivant. Au total on a $9*\log(n)$ opérations élémentaires. La complexité est en $O(\log(n))$.

1. **Suite de Fibonacci (**Exercice laissé aux lecteurs motivés.**)**

Voici deux implémentations différentes de la suite de Fibonacci, analysez leurs complexités et comparez-les. 

```python
def fibonacci_v1(n:int):
    
    if n==0: return 0
    res1=0
    res2=1
    
    for i in range(1,n):
        tmp = res1+res2
        res1 = res2
        res2 = tmp
    return res2
```

```python
def fibonacci_v2(n:int):
    
    if n==0: return 0
    if n==1: return 1
    
    res1= fibonacci_v2(n-1)
    res2= fibonacci_v2(n-2)
    return res1 + res2
```

### Complexité dans le Machine Learning

Nous allons dans un premier temps étudier la complexité des deux algorithmes de base en Machine Learning que nous connaissons bien, du moins pour les initiés et les avertis. Pour mieux les comprendre, nous allons les recoder à partir de zéro en utilisant la librairie numpy, ensuite estimer leurs complexités pour l’entrainement et l’inférence.

Supposons que la matrice $X$ représentant le jeu de données, elle est de taille $n*p$ où $n$ est le nombre d’observations dans le jeu de données et $p$ le nombre de caractéristiques (features).  Notons  $n>>p$, ce qui est généralement le cas pour les jeux de données où la dimension des échantillons peut être négligeable par rapport à la taille du jeu de données. Les opérations élémentaires ici seront les multiplications des entiers et les allocations mémoires pour les variables pour nos calculs de complexités. La matrice $y$ est de taille $n$ également et représente les étiquettes pour chaque donnée (ligne de la matrice $X$).

1. **Cas d’une régression Linéaire ( Linear Regression)**

```python
import numpy as np

class LinearRegression():
    """ Linear regression model using normal equation """
    def __init__(self) -> None:
        self.weights = None

    def fit(self, X, y):
        if self.weights is not None:
            raise ValueError("Model is already fit")
        
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        invers_X = np.linalg.inv(X.T.dot(X))
        self.weights = invers_X.dot(X.T).dot(y)

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model is not fit yet")
        if self.weights.shape[0] != X.shape[1] + 1:
            raise ValueError("Input shape is not compatible with model")
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return self.weights.T.dot(X)
```

Analysons la complexité temporelle et spatiale de ce code:

**Complexité temporelle**

1. La méthode __**init**__

Cette méthode ne fait pratiquement rien en termes d’opérations, sa complexité est constante $O(1)$.

1. La méthode **fit** (la méthode qui entraine le modèle)

On a une première multiplication $X'\times X$, pour chaque élément de la nouvelle matrice il faut $p$ multiplications et $p-1$ additions. Pour donc la nouvelle matrice qui contiendra $np$ éléments, il faut $(p + (p-1))\times np \approx 2p^2n$, donc la complexité est **$O(p^2n)$.**    

L’inverse matriciel avec np.linalg.inv de la bibliothèque numpy a une complexité $O(n^3)$ pour une matrice carré de taille $n$. Je laisserai au lecteur le soins de vérifier cette complexité.   

Ensuite, on a une multiplication d’une matrice et d’un vecteur. Comme pour la première multiplication, pour chaque élément on a $2p-1$ opérations, donc au total $p(2p-1)$, la complexité est donc $O(p^2)$.

En faisant la somme de ces différentes complexités, la complexité est de $O(p^2 + p^2n + p^3) = O(p^2n + p^3)$.

1. La méthode **predict** ( méthode d’inférence)

Dans cette méthode, on a une seule multiplication entre le vecteur d’une taille $p$ contenant les poids et une matrice de taille $np$ contenant les features pour chaque exemple. Pour calculer la valeur finale, on aura $n²p$ opérations, la complexité est $O(p²n)$.

**Complexité spatiale**

Dans la méthode init on ne stocke rien, si ce n’est la référence None à la variable. On a une complexité constante $O(1)$. Dans la méthode fit dans un premier temps, on stocke la matrice X de taille $n \times p$  représentant le jeu de données à laquelle on rajoute un vecteur unitaire, donc on a une allocation mémoire de $(n \times p )+ p$. La variable invers_X est de meme taille et occupe donc un espace mémoire de  $n \times p$. La variable weight est un vecteur de taille $p$ et occupe donc un espace mémoire, la compléxité est alors $O(np + p)= O(np)$.

1. **Cas de la régression Logistique (Logistic Regression)**

```python
import numpy as np

class logistic_regression():

    def __init__(self, lr:int=0.001, n_iter:int=1000,epsilon=1e-9, treshold=.5):
        """ initialise all parameters """
        
        self.lr = lr
        self.n_iter = n_iter
        self.epsilon=epsilon
        self.treshold=treshold
        self.print_loss=True

        # model parameters
        self.w = None
        self.b = None
        
    def _sigmoid_function(self, x:np.array)-> np.array:
        """ Compute the sigmoid value of a vector x """
        
        return 1/(1+np.exp(-x))

    def feed_forward(self,X:np.array) -> np.array:
        """Compute the ouput probability"""

        z = X@self.w + self.b
        return self._sigmoid_function(z)

    def compute_loss(self, y_true:np.array, y_pred:np.array):
        """ Compute the binary cross entropy """
        
        #epsilon if y_pred equal zero or one
        res1 = y_true*np.log(y_pred+self.epsilon)
        res2 = (1-y_true)*np.log(1-y_pred+self.epsilon)        
        
        return -np.mean(res1+res2)

    def fit(self, X:np.array, y:np.array) -> None:
        """ Fit the model weight using gradient descent algorithm """

        self.w, self.b = np.zeros(X.shape[1]), 0

        for i in range(self.n_iter):

            output_prob = self.feed_forward(X)
            dz =  output_prob - y
            dw = np.mean(X.T@dz)
            db = np.mean(dz)

            self.w = self.w - self.lr*dw
            self.b = self.b - self.lr*db

    def predict(self,X:np.array)-> np.array:
        
        if self.w is None or self.b is None:
            raise('Model has not been fitted yet, please run fit first !')
        output_prob = self.feed_forward(X)
        output_lab = [1 if pred>self.treshold else 0 for pred in output_prob]

        return output_lab

```

**Complexité temporelle**

La méthode init a une complexité temporelle constante, elle ne stocke que des constantes indépendantes des entrées et n’effectue aucun calcul. 

La méthode sigmoid effectue une opération d’exp, en considérant cette opération élémentaire pour chaque élément du vecteur de taille $p$, on a donc au total $p$ opérations, donc une complexité en $O(p)$. 

La méthode feed_forward fait une multiplication matricielle dont la compléxité est $O(p²n)$, une addition matricielle dont la compléxité est $O(p)$, et enfin la méthode sigmoid dont la complexite est $O(p)$. En addition ces compléxité est en tenant compte de l’ordre de grandeur; cette méthode a une compléxité $O(p²n)$.  

La méthode compute_loss prend en entrée deux vecteurs de même taille $p$. Il y a quatre opérations  addition/soustraction, et deux multiplications vectorielles (élément par élément) entre vecteurs dont la complexité est $O(p)$. L’opération np.log est constante, sur un vecteur de taille p, sa complexité est $O(p)$. En additionnant ces complexités, on obtient une complexité $O(p)$. 

La méthode fit, la première ligne est une affectation pas de calcul. Examinons un tour de boucle :

La première instruction est un appel à feed_forward dont la complexité est en $O(p²n)$. Ensuite, une multiplication matricielle (matrice-vecteur) dont la complexité est la même que la méthode feed_forward. La fonction np.mean de numpy a une complexité linéaire $O(p)$, les autres opérations d’additions et multiplications sont constantes. Pour la mise à jour de w, on a $O(p)$ en termes de complexité. En additionnant, on obtient une complexité $O(p²n)$. 

Pour un nombre $m$ d’itérations, on aura $m*O(p²n)$, et donc la complexité de cette méthode fit est de $O(mnp²)$. 

Pour la méthode predict, on a un appel à feed_forward et une boucle sur chaque élément du vecteur de probabilité de taille $p$, donc cette méthode a une complexité $O(p²n)$. 

**Compléxité spatiale**

La méthode **init** a une complexité constante, elle ne stocke que des constantes. La méthode _sigmoid_function a besoin de stocker le résultat, dont un vecteur de taille $p$. La complexité est en $O(p)$. Pour la méthode feed_forward, il faut stocker le vecteur z de taille $p$, et le résultat qui est de taille $p$, donc $2p$, la complexité est de $O(p)$. La méthode compute_loss stocke les deux variables res1 et res2, ainsi que le résultat, chacune a besoin d’une allocation mémoire de $p$, donc une complexité de $O(p)$. 

La méthode **fit** avant ****la boucle ne stocke que deux variables dont l’espace mémoire est de taille p. Les six variables dans la boucle ont besoin chacune d’un espace mémoire de p, donc la complexité pour un tour de boucle est $O(p)$, donc pour m tour de boucle, on aura une complexité finale $O(mp)$. Easy !

**Conclusion**

L’étude de la complexité ne peut être réduite à ce blog. Le but ici était de vous donner quelques éléments clés pour évaluer la complexité d’un algorithme de machine learning ou non. La complexité ne remplace pas le benchmark, mais c’est un outil qui permet d’avoir une idée du comportement d’un algorithme indépendamment du langage d’implémentation et des détails d’optimisation liés à ce langage. La constante que cache la notation  peut être trompeuse, puisqu’elle peut être très grande et qu’en pratique, un algorithme de complexité polynomiale soit plus rapide qu’un algorithme de complexité logarithmique. Le Quick sort n’est pas le meilleur algorithme de tri en ce qui concerne la complexité, mais il est tout de même le meilleur en pratique et reste un choix par défaut.