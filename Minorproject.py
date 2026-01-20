#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score 
from sklearn.ensemble import RandomForestClassifier


# In[3]:


data= pd.read_csv('Synthetic_Financial_datasets_log.csv')
data.head()


# In[3]:


data['type'].unique()


# In[4]:


data.describe()


# In[5]:


data.info()


# In[6]:


data.isnull().dropna().sum()


# In[7]:


data.shape[0]


# In[8]:


data['isFraud'].nunique()


# In[9]:


data['isFraud'].unique()


# In[10]:


data['isFraud'].value_counts()


# In[11]:


print("No. of valid transactions ",data.isFraud.value_counts()[0])
print("No. of fraud transactions ",data.isFraud.value_counts()[1])


# In[12]:


data['isFraud'].value_counts().plot.bar(color=['blue','red'])
plt.show()


# In[13]:


print("No of transactions Flagged valid :",data['isFlaggedFraud'].value_counts()[0])
print("No of transactions Flagged fraud :",data['isFlaggedFraud'].value_counts()[1])


# In[4]:


#checking for any error at origin and destination
data['error_org']= ((data['oldbalanceOrg']-data['amount']) != data['newbalanceOrig']).astype(int)
data['error_dst']= ((data['oldbalanceDest']+data['amount']) != data['newbalanceDest']).astype(int)

error_percent_org= round(data['error_org'].value_counts()[1]/data.shape[0] * 100, 2) 
error_percent_dest= round(data['error_dst'].value_counts()[1]/data.shape[0] * 100, 2)
print("Error at Origin:",error_percent_org,"%")
print("Error at Destination:",error_percent_dest,"%")


# *This shows that the fraud occurs both at the origin and the destination.

# In[15]:


print("Transactions less than Amount 0:")
print(len(data[data.amount<=0]))
print("Type of transaction:")
print(data[data.amount<=0]['type'].value_counts().index[0])
print("Are all these marked as Fraud Transactions?")
data[data.amount<=0]['isFraud'].value_counts()[1] == len(data[data.amount<=0])


# In[16]:


data_temp = data[data.isFlaggedFraud==1]
print("Minumum amount transfered in FlaggedFraud transactions")
print("\t",data_temp.amount.min())

print("Maximum amount transfered in FlaggedFraud transactions")
print("\t",data_temp.amount.max())


# In[17]:


min_val = data_temp.amount.min()
max_val = data_temp.amount.max()

plt.bar(['Min Amount', 'Max Amount'], [min_val, max_val])
plt.title('Flagged Fraud Transactions Amounts')
plt.ylabel('Amount')

plt.show()


# In[18]:


print("Transactions with amount less than or equal to 0:")
print(len(data[data.amount<=0]))
print("Type of these transactions:")
print(data[data.amount<=0]['type'].value_counts().index[0])
print("Are these transactions marked fraud?")
data[data.amount<=0]['isFraud'].value_counts()[1] == len(data[data.amount<=0])


# In[19]:


data_temp = data[data.isFlaggedFraud==1]
print("How many frauds transactions are Flagged?:")
print("\t",len(data_temp))

print("What type of transactions are they?")
print("\t",data_temp['type'].value_counts().index[0])

print("Are all these flagged also marked as Fraud Transactions?")
print("\t",data_temp['isFraud'].value_counts()[1] == len(data_temp))

print("Minumum amount transfered in these transactions")
print("\t",data_temp.amount.min())

print("Maximum amount transfered in these transactions")
print("\t",data_temp.amount.max())


# These means most of the fraud activity happens during CASH_OUT and TRANSFER

# In[5]:


data = data.loc[(data['type'].isin(['TRANSFER', 'CASH_OUT']))]
data.head()


# In[21]:


plt.figure(figsize=(8,6))
plt.pie(data.type.value_counts().values,labels=data.type.value_counts().index,  autopct='%.0f%%')
plt.title("Transaction Type")
plt.show()


# In[22]:


d = data.groupby('type')['amount'].sum()
plt.figure(figsize=(8,6))
ax = sns.barplot(x=d.index,y=d.values)
for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x()+0.24, p.get_height()*1.01))
    
plt.title("Total amount in each transaction type")
plt.yticks([])
plt.xlabel("Transaction Type")
plt.show()


# In[6]:


data.drop(['step','type','nameOrig','nameDest','error_org','error_dst','isFlaggedFraud'],axis=1,inplace=True)
data.head()


# In[7]:


#standardizing the values to be between 0 to 1.
ss = StandardScaler()

data.amount         = ss.fit_transform(data[['amount']])
data.oldbalanceOrg  = ss.fit_transform(data[['oldbalanceOrg']])
data.oldbalanceDest = ss.fit_transform(data[['oldbalanceDest']])
data.newbalanceOrig = ss.fit_transform(data[['newbalanceOrig']])
data.newbalanceDest = ss.fit_transform(data[['newbalanceDest']])


# In[8]:


x = data.drop(["isFraud"],axis=1)
y = data.isFraud
x_train, x_test, y_train, y_test = train_test_split(x, y,stratify=y)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[9]:


def conf_matrix(y_test, pred_test):    
    
    # Creating a confusion matrix
    con_mat = confusion_matrix(y_test, pred_test)
    con_mat = pd.DataFrame(con_mat, range(2), range(2))
   
    #Ploting the confusion matrix
    
    plt.figure(figsize=(6,6))
    plt.title("Confusion Matrix")
    sns.set(font_scale=1.5) 
    sns.heatmap(con_mat, annot=True, annot_kws={"size": 16}, fmt='g', cmap='crest', cbar=False)


# In[28]:


lr = LogisticRegression(solver='newton-cg')
lr.fit(x_train, y_train)

lr_pred = lr.predict(x_test)

print("Classes the model predicted:",np.unique( lr_pred ))
print("Numbers in each class:\t\t","0 :",len(lr_pred[lr_pred==0]))
print("\t\t 1 :",len(lr_pred[lr_pred==1]))

f1score = f1_score(y_test, lr_pred)
print('f1 score:', f1score)

conf_matrix(y_test, lr_pred)
 
acc_lr= accuracy_score(y_test, lr_pred)
print("Accuracy of this model:", acc_lr)


# In[29]:


#since the data is extremely imbalanced
from sklearn.utils import resample
n = data.isFraud.value_counts()[0]

# Separate majority and minority classes
df_majority = data[data.isFraud==0]
df_minority = data[data.isFraud==1]

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=n,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
print("The new class count are :")
df_upsampled.isFraud.value_counts()


# In[30]:


x = df_upsampled.drop(["isFraud"],axis = 1)
y = df_upsampled.isFraud
x_train, x_test, y_train, y_test = train_test_split(x, y)

lr = LogisticRegression(solver='newton-cg')
lr.fit(x_train, y_train)

# Predicting on the test data
up_scale_pred = lr.predict(x_test)

#Calculating and printing the f1 score 
f1up_scale_pred = f1_score(y_test, up_scale_pred)
print('f1 score for the testing data:\t', f1up_scale_pred)

#Calling function 
conf_matrix(y_test,up_scale_pred)

acc_up_scale=accuracy_score(y_test, up_scale_pred)
print("Accuracy of thie model:\t\t",acc_up_scale)


# In[31]:


n = data.isFraud.value_counts()[1]

# Separate majority and minority classes

df_majority = data[data.isFraud==0]
df_minority = data[data.isFraud==1]

 
# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=n,     # to match minority class
                                 random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
 
print("The new class count are:")
print(df_downsampled.isFraud.value_counts())


# In[32]:


y = df_downsampled.isFraud
x = df_downsampled.drop(['isFraud'], axis=1)
 
# Train model
lr = LogisticRegression().fit(x, y)
 
# Predict on training set
down_scale_pred = lr.predict(x)
 
print("How many class does the model predict?",np.unique( down_scale_pred ))
print("Count in each class:\t\t\t","0 :",len(down_scale_pred[down_scale_pred==0]))
print("\t\t\t\t\t 1 :",len(down_scale_pred[down_scale_pred==1]))

#Calculating and printing the f1 score 
f1down_scale_pred = f1_score(y, down_scale_pred)
print('f1 score for the testing data:\t\t', f1down_scale_pred)

conf_matrix(y, down_scale_pred)
      
acc_down_scale=accuracy_score(y, down_scale_pred) 
print("Accuracy of the model:\t\t\t", acc_down_scale)


# In[11]:


y = data.isFraud
x = data.drop(['isFraud'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y)

# Train model
rfc = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    min_samples_leaf=5,
    n_jobs=-1,
random_state=42
)
rfc.fit(x_train, y_train)
# Predict on training set
rfc_pred = rfc.predict(x_test)
prob_y = rfc.predict_proba(x_test)[:,1]

print("AUROC:\t\t\t",roc_auc_score(y_test, prob_y))

f1_rfc = f1_score(y_test, rfc_pred)
print('f1 score:\t\t', f1_rfc)

conf_matrix(y_test, rfc_pred)

acc_rfc=accuracy_score(y_test, rfc_pred) 
print("Accuracy of the model:\t", acc_rfc)


# In[3]:


import pandas as pd
import joblib

# Saving the trained model in .joblib format for further evaluation
data_full = pd.read_csv("Synthetic_Financial_datasets_log.csv")

# Keep only TRANSFER and CASH_OUT as in your EDA
data_model = data_full[data_full["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

# Feature Engineering 
data_model["error_org"] = (data_model["oldbalanceOrg"] - data_model["amount"] != data_model["newbalanceOrig"]).astype(int)

data_model["error_dst"] = (data_model["oldbalanceDest"] + data_model["amount"] != data_model["newbalanceDest"]).astype(int)

# Final features and label
feature_cols = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "error_org",
    "error_dst",
]
target_col = "isFraud"

X = data_model[feature_cols]
y = data_model[target_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

#Using RandomForest model to detect fraud
rfc = RandomForestClassifier(
    n_estimators=80,
    max_depth=12,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
)

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
y_prob = rfc.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("AUROC:", roc_auc_score(y_test, y_prob))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Saving model for live streaming
joblib.dump(rfc, "fraud_model.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
print("Saved fraud_model.pkl and feature_cols.pkl")


# In[18]:


get_ipython().run_cell_magic('writefile', 'kafka_producer.py', 'from kafka import KafkaProducer\nimport json\nimport time\nimport random\n\nproducer = KafkaProducer(\n    bootstrap_servers="localhost:9092",\n    value_serializer=lambda v: json.dumps(v).encode("utf-8"),\n)\n\ndata = pd.read_csv("Synthetic_Financial_datasets_log.csv")\ndata = data[data["type"].isin(["TRANSFER", "CASH_OUT"])].copy()\n\nemails = ["user1@example.com", "user2@example.com", "user3@example.com"]\n\nfor _, row in data.sample(500, random_state=42).iterrows():\n    msg = {\n        "amount": float(row["amount"]),\n        "oldbalanceOrg": float(row["oldbalanceOrg"]),\n        "newbalanceOrig": float(row["newbalanceOrig"]),\n        "oldbalanceDest": float(row["oldbalanceDest"]),\n        "newbalanceDest": float(row["newbalanceDest"]),\n        "email": random.choice(emails),\n        "isFraudLabel": int(row["isFraud"]),\n    }\n    producer.send("transactions", value=msg)\n    print("Sent:", msg)\n   \xa0time.sleep(0.5)\n')


# In[43]:


get_ipython().run_cell_magic('writefile', 'fraud_consumer.py', 'from kafka import KafkaConsumer\nimport json\nimport joblib\nimport numpy as np\nimport smtplib\nfrom email.mime_text import MIMEText\n\n# Load model + feature columns\nmodel = joblib.load("fraud_model.pkl")\nfeature_cols = joblib.load("feature_cols.pkl")\n\n# setting gmail \nSENDER_EMAIL = "ishvee09@gmail.com"\nSENDER_APP_PASSWORD = "gtbv ejut zjvh zsrz"  # Gmail App Password\n\ndef send_fraud_email(to_email, amount, prob):\n    if not to_email:\n        return\n    body = (\n        " Suspicious transaction detected.\\n\\n"\n        f"Amount: {amount}\\n"\n        f"Model fraud probability: {prob:.2f}\\n\\n"\n        "If this was not you, please contact the bank immediately."\n    )\n    msg = MIMEText(body)\n    msg["Subject"] = "Fraud Alert"\n    msg["From"] = SENDER_EMAIL\n    msg["To"] = to_email\n\n    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:\n        server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)\n        server.send_message(msg)\n\nconsumer = KafkaConsumer(\n    "transactions",\n    bootstrap_servers="localhost:9092",\n    value_deserializer=lambda v: json.loads(v.decode("utf-8")),\n    auto_offset_reset="latest",\n    enable_auto_commit=True,\n)\n\nprint("Listening for transactions...")\n\nfor msg in consumer:\n    data = msg.value\n\n    amount = float(data["amount"])\n    old_org = float(data["oldbalanceOrg"])\n    new_org = float(data["newbalanceOrig"])\n    old_dst = float(data["oldbalanceDest"])\n    new_dst = float(data["newbalanceDest"])\n\n    error_org = int(old_org - amount != new_org)\n    error_dst = int(old_dst + amount != new_dst)\n\n    row_dict = {\n        "amount": amount,\n        "oldbalanceOrg": old_org,\n        "newbalanceOrig": new_org,\n        "oldbalanceDest": old_dst,\n        "newbalanceDest": new_dst,\n        "error_org": error_org,\n        "error_dst": error_dst,\n    }\n\n    X = np.array([[row_dict[c] for c in feature_cols]])\n    prob = model.predict_proba(X)[0, 1]\n    pred = int(prob >= 0.5)\n\n    print("Received:", data, "=> prob:", prob, "pred:", pred)\n\n    if pred == 1:\n        send_fraud_email(data["email"], amount, prob)\n        print("Fraud email sent to",\xa0data["email"])\n')


# In[48]:


SENDER_EMAIL = "ishvee09@gmail.com"
SENDER_APP_PASSWORD = "gtbv ejut zjvh zsrz"  


# In[49]:


def send_fraud_email(to_email, amount, prob):
    if not to_email:
        return

    body = (
        "Suspicious transaction detected!\n\n"
        f"Amount: {amount}\n"
        f"Fraud Probability: {prob:.2f}\n\n"
        "If this was not you, please contact the bank immediately."
    )

    msg = MIMEText(body)
    msg["Subject"] = "Fraud Alert"
    msg["From"] = SENDER_EMAIL
    msg["To"] = to_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(" Fraud alert email sent!")
    except Exception as e:
        print(" Email sending failed:",e)


# In[45]:


from kafka import KafkaConsumer
import json
consumer = KafkaConsumer(
    "fraud_transactions",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)

print(" Kafka Consumer started... Waiting for messages...")


# In[46]:


#saving the final model
import joblib

model = joblib.load("fraud_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")

print("Model and feature columns loaded!")


# In[ ]:


for message in consumer:
    txn = message.value

    # Building feature vector in correct order
    feature_values = [txn[col] for col in feature_cols]
    X_live = np.array(feature_values).reshape(1, -1)

    # Predicting fraud probability
    prob = model.predict_proba(X_live)[0, 1]

    print("Transaction received:", txn)
    print("Fraud probability:", prob)

    
    if prob > 0.000001 :
        send_fraud_email(
            to_email=txn.get("email"),
            amount=txn.get("amount"),
            prob=prob)


# In[ ]:




