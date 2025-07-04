import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Carica dati
df = pd.read_csv("G2E2_clienti_churn_molti_dati.csv")
train = df.iloc[:45]
test = df.iloc[45:]

X_train = train[["num_reclami", "mesi_ultimo_acquisto", "spesa_mensile", "cliente_vip"]]
y_train = train["churn"]
X_test = test[["num_reclami", "mesi_ultimo_acquisto", "spesa_mensile", "cliente_vip"]]
y_test = test["churn"]

# Addestra XGBoost
model_xgb = XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.01, use_label_encoder=False, eval_metric='logloss', random_state=42)
model_xgb.fit(X_train, y_train)

# Predizioni sul test set
y_pred = model_xgb.predict(X_test)

# Tabella risultati
output = test
output["churn_predetto_xgb"] = y_pred
output.to_csv("G2E3_risultati_churn_xgboost_tuning.csv", index=False)
print("Risultati salvati in 'G2E3_risultati_churn_xgboost_tuning.csv'")

# Metriche di valutazione
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))
