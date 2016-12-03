import matplotlib.pyplot as plt
results = [
(0.66666666666666663, 0.5, 0.57142857142857151, None),
(0.5714285714285714, 0.59999999999999998, 0.58536585365853655, None),
(0.52173913043478259, 0.59999999999999998, 0.55813953488372092, None),
(0.5, 0.40000000000000002, 0.44444444444444448, None),
(0.44827586206896552, 0.65000000000000002, 0.53061224489795922, None),
(0.68421052631578949, 0.65000000000000002, 0.66666666666666674, None),
(0.625, 0.75, 0.68181818181818177, None),
(0.69999999999999996, 0.69999999999999996, 0.69999999999999996, None),
(0.51851851851851849, 0.69999999999999996, 0.59574468085106391, None),
(0.51724137931034486, 0.75, 0.6122448979591838, None),
(0.39130434782608697, 0.45000000000000001, 0.41860465116279072, None),
(0.61904761904761907, 0.65000000000000002, 0.63414634146341464, None),
(0.59999999999999998, 0.75, 0.66666666666666652, None),
(0.68421052631578949, 0.65000000000000002, 0.66666666666666674, None),
(0.81818181818181823, 0.45000000000000001, 0.58064516129032262, None),
(0.61111111111111116, 0.55000000000000004, 0.57894736842105265, None),
(0.51851851851851849, 0.69999999999999996, 0.59574468085106391, None),
(0.60869565217391308, 0.69999999999999996, 0.65116279069767435, None),
(0.44444444444444442, 0.59999999999999998, 0.5106382978723405, None),
(0.64000000000000001, 0.80000000000000004, 0.71111111111111114, None)]



f_scores = sum([x[2] for x in results]) / len(results)
print "Average F-Score",f_scores
precisions = [x[0] for x in results]
recalls = [x[1] for x in results]
plt.clf()
plt.title("Scatter plot of Precision and Recalls for various users")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.scatter(precisions,recalls)
plt.savefig("precision_recall_curve.png")