from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
A = [[1,0,0,0]]
B = [[1,0,0,0]]
print precision_score(A, B)