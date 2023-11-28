#Régression linéaire

# importations
import numpy as np
import numpy.linalg as la
import math
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
from scipy.stats import linregress

t = [0, 0.5, 1, 1.5 , 2 , 3, 5, 7.5, 10, 15, 20, 24]
c = [ math.log(23.8), math.log(22.1), math.log(20.5), math.log(19),
 math.log(17.6), math.log(15.2), math.log(11.3), math.log(7.82), math.log(5.39),
  math.log(2.57), math.log(1.22), math.log(0.68)]

#print(c)   
#c = ln(c)

##TEST 1 :
# tracer le graphique
plt.scatter(t,c,s=5)
plt.xlabel('x')
plt.ylabel('y')
#plt.show()

## TEST 2 :
def regLin(x, y):
    """
    Ajuste une droite d'équation a*x + b sur les points (x, y) par la méthode
    des moindres carrés.
    Args :
        * x (list): valeurs de x
        * y (list): valeurs de y
    Return:
        * a (float): pente de la droite
        * b (float): ordonnée à l'origine
    """
    # initialisation des sommes
    x_sum = 0.
    x2_sum = 0.
    y_sum = 0.
    xy_sum = 0.
    # calcul des sommes
    for xi, yi in zip(x, y):
        x_sum += xi
        x2_sum += xi**2
        y_sum += yi
        xy_sum += xi * yi
    # nombre de points
    npoints = len(x)
    # calcul des paramétras
    a = (npoints * xy_sum - x_sum * y_sum) / (npoints * x2_sum - x_sum**2)
    b = (x2_sum * y_sum - x_sum * xy_sum) / (npoints * x2_sum - x_sum**2)
    # renvoie des parametres
    return a, b
   
#print(regLin(t,c))
#a,b=regLin(t,c)
(a,b,rho,p,stderr)=linregress(t,c)

plt.plot(t, c, "bo", label="DONNEES") # les points (x, y) representes par des points - , color = "pink" / Couleur déterminée par "bo"
plt.plot( # droite de regression
    [t[0], t[-1]],                  # valeurs de x
    [a * t[0] + b, a * t[-1] + b],  # valeurs de y
    "r-",                           # couleur rouge avec un trait continu ou color="#FFDFD3"
    label="REGRESSION", )             # legende
plt.xlabel("x") # nom de l'axe x
plt.ylabel("y") # nom de l'axe y
plt.xlim(-1, 25) # échelle axe x
plt.ylim(-1,6)
plt.legend() # la legende
plt.title("Regression Lineaire") # titre de graphique
#Résultats : (-0.9689814524630331, 19.491986666286785)
plt.show()

## Test 3
#X = np.linalg.solve(t,c)

#print(a,b)
#print(b)
#print(math.exp(b))
#c0 = math.exp(b)
#print("c0=", c0, "k=", a)
#c0= 23.743333042478067 k= -0.1482142234460459
#ln(c0) = 3.1673017770721765


#x, res, r, s= np.linalg.lstsq(A,b)

A= np.array([[1,0], [1,0.5], [1,1], [1,1.5] , [1,2] ,[ 1,3],
 [1,5], [1,7.5], [1,10], [1,15], [1,20], [1,24]])
#print(A)

b= np.array([c])
#print(b)
bT = b.transpose()
#print(bT)
AT= A.transpose()
#print(AT)

AT_A = np.dot(AT, A)
AT_b = np.dot(AT, bT)

#print(AT_A, AT_b)

#Alpha= la.solve(AT_A, AT_b)
Alpha = np.array([3.1673017770721765, -0.1482142234460459])


#Alpha= (ln(c0), k)

norme_b1 = 0
for i in  c :
      norme_b1 += i**2
      

norme_b = np.sqrt(norme_b1)
#print("norme_b = ",norme_b)

J_Alpha = np.dot(A,Alpha) - c
print(J_Alpha)
norme_J = 0
for i in  J_Alpha :
      norme_J += i**2

      
Eps = np.sqrt(norme_J)/norme_b
print(Eps)

#psilon = 0.0009643800858537833
