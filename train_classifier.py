import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def pad_sequences(data, maxlen=None, dtype='float32', padding='post', value=0.0):
    lengths = [len(seq) for seq in data]
    maxlen = maxlen or max(lengths)

    padded_data = np.full((len(data), maxlen), value, dtype=dtype)
    for i, seq in enumerate(data):
        if len(seq) > maxlen:
            padded_data[i, :maxlen] = seq[:maxlen]
        else:
            padded_data[i, :len(seq)] = seq
    return padded_data

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Pad the sequences
data_padded = pad_sequences(data)

x_train, x_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=True, stratify=labels)

# Proceed with the rest of the code
model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)