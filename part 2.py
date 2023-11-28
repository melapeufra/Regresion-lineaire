import matplotlib.pyplot as plt
import numpy as np
#from sklearn.metrics import r2_score#pour calculer le coeff R2

fig=plt.figure(figsize=(7, 9), tight_layout=True)
#plt.suptitle("Exemples de régressions de polynômes")
ax1=plt.subplot(211)
plt.title("Polynôme degré 3")
#1er exemple ---------------------------------------------------------------------------
x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]
y = [2313,2653,2548,3499,3593,5235,3527,4208,5324,5988,6554,5560,
4783,5240,5198,6202,5907,5973,5215,4047,4053,4783,4669,4585,4807,4318,3599,3037,3834,
4204,3950,4697,4094,3153,2973,2666,3787,3494,3492,3047,2251,2727,3370,2644]

mymodel_4 = np.poly1d(np.polyfit(x, y, 4))
mymodel_3 = np.poly1d(np.polyfit(x, y, 3)) #Donne l'équation
mymodel_2 = np.poly1d(np.polyfit(x, y, 2))
#myline = np.linspace(1, 22, 200) --> Pareil

plt.scatter(x, y)
plt.xlabel("x") # nom de l'axe x
plt.ylabel("y") # nom de l'axe y
plt.xlim(-1, 50) # échelle axe x
plt.ylim(-100,7000)
#plt.plot(myline, mymodel(myline)) --> ??
plt.text(12, -530, mymodel_4)
plt.text(12, 30, mymodel_3)
plt.text(12, 530, mymodel_2)
#plt.text(12, 300, '{:.5f}'.format(r2_score(y, mymodel(x))))

#ax2=plt.subplot(212)
#plt.title("polynôme degré 2")

P_2=np.polyfit(x, y, 2)
P_3=np.polyfit(x, y, 3)
P_4=np.polyfit(x, y, 4)
#P est une liste des valeurs a,b et c du polynome ax**2+bx+c=0
monmodele_2=np.poly1d(P_2)
monmodele_3=np.poly1d(P_3)
monmodele_4=np.poly1d(P_4)
#plt.scatter(x, y)
plt.plot(x, monmodele_2(x),label="Regression degre 2", color="red")
plt.plot(x, monmodele_3(x),label="Regression degre 3", color="green")
plt.plot(x, monmodele_4(x),label="Regression degre 4", color="yellow")
#la valeur de R2 se récupère avec r2_score(y,monmodele(x))
#plt.text(0.5, 0.5, '{:.5f}'.format(r2_score(y, monmodele(x))))
#plt.text(0.4, 300, monmodele)
#------------------------------------------------------------------------
fig.savefig("RegressionPolynomes2-3.png", dpi=250)
plt.legend()
plt.show()

A= np.array([[1,0,0**2],[1,1,1**2],[1,2,2**2],[1,3,3**2],[1,4,4**2],[1,5,5**2],[1,6,6**2],[1,7,7**2],[1,8,8**2],[1,9,9**2],[1,10,10**2],[1,11,11*2],[1,12,12**2],[1,13,13**2],[1,14,14**2],[1,15,15**2],[1,16,16**2],[1,17,17**2],[1,18,18**2],[1,19,19**2],[1,20,20**2],[1,21,21**2],[1,22,22**2],[1,23,23**2],[1,24,24**2],[1,25,25**2],[1,26,26**2],[1,27,27**2],[1,28,28**2],[1,29,29**2],[1,30,30**2],[1,31,31**2],[1,32,32**2],[1,33,33**2],[1,34,34**2],[1,35,35**2],[1,36,36**2],[1,37,37**2],[1,38,38**2],[1,39,39**2],[1,40,40**2],[1,41,41**2],[1,42,42**2],[1,43,43**2]])

print(A)
