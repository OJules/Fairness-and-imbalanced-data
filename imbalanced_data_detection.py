#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install imbalanced-learn


# In[3]:


# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Chargement des données
print("Chargement des données...")
data = fetch_openml('creditcard')
X = pd.DataFrame(data.data)
y = pd.Series(data.target)

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=42)

# Standardisation uniquement de Amount
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Standardisation de Amount
X_train_scaled['Amount'] = scaler.fit_transform(X_train[['Amount']])
X_test_scaled['Amount'] = scaler.transform(X_test[['Amount']])

# Affichage des statistiques avant/après standardisation pour Amount
print("\nStatistiques pour Amount:")
print("Avant standardisation:")
print(X_train['Amount'].describe())
print("\nAprès standardisation:")
print(X_train_scaled['Amount'].describe())

# Application de SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Vérification de la distribution après SMOTE
print("\nDistribution après SMOTE :")
print(pd.Series(y_train_balanced).value_counts(normalize=True))


# In[4]:


# Création du pipeline avec LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Conversion des étiquettes en numériques
y_train_balanced = y_train_balanced.astype(int)
y_test = y_test.astype(int)

# Création du modèle
model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

# Entraînement du modèle
model.fit(X_train_balanced, y_train_balanced)

# Prédictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Évaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Courbe ROC
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[6]:


get_ipython().system('pip install xgboost')


# In[8]:


# Import des modèles
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Liste des modèles à tester
models = {
    'Random Forest': RandomForestClassifier(
        class_weight='balanced',
        n_estimators=100,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        random_state=42
    ),
    'XGBoost': XGBClassifier(
        scale_pos_weight=1,  # pour le déséquilibre
        random_state=42
    )
}

# Évaluation de chaque modèle
for name, model in models.items():
    print(f"\nÉvaluation du modèle : {name}")
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Matrice de confusion
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de Confusion - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


# In[9]:


from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# Création du pipeline
pipeline = Pipeline([
    ('feature_selection', SelectKBest(f_classif, k=10)),  # Sélection des 10 meilleures features
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Entraînement
pipeline.fit(X_train_balanced, y_train_balanced)

# Prédictions
y_test_preds = pipeline.predict(X_test)
y_test_probas = pipeline.predict_proba(X_test)

# Évaluation
print('Métriques d\'évaluation:')
print('F1 score:', f1_score(y_test, y_test_preds))
print('AUROC:', roc_auc_score(y_test, y_test_probas[:, 1]))
print('AUPRC:', average_precision_score(y_test, y_test_probas[:, 1]))


# In[10]:


# Pour tracer ROC et PRC
from sklearn.metrics import roc_curve, precision_recall_curve

# Calcul des points pour les courbes
fpr, tpr, thresholds = roc_curve(y_test, y_test_probas[:, 1])
precision, recall, thresholds = precision_recall_curve(y_test, y_test_probas[:, 1])

# Création des visualisations
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Courbe ROC
axes[0].plot(fpr, tpr)
axes[0].set_xlabel('FPR (Taux de faux positifs)')
axes[0].set_ylabel('TPR (Taux de vrais positifs)')
axes[0].set_title('Courbe ROC')

# Courbe PRC
axes[1].plot(recall, precision)
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Courbe PRC')

# Ajustement des limites
axes[0].set_xlim([-.05, 1.05])
axes[0].set_ylim([-.05, 1.05])
axes[1].set_xlim([-.05, 1.05])
axes[1].set_ylim([-.05, 1.05])

fig.tight_layout()
plt.show()

