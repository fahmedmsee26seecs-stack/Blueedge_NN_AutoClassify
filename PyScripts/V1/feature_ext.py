import re, math
import numpy as np
from collections import Counter

PATTERNS = [
    r"^[\w.+-]+@[\w-]+\.[\w.]+$",                    # email
    r"^\+?[\d\s().-]{7,15}$",                        # phone
    r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}$",            # date
    r"^\d{3}-\d{2}-\d{4}$",                          # ssn
    r"^(https?|ftp)://\S+$|^www\.\S+$",              # url
    r"^\d+$",                                        # numeric id
    r"^[$€£¥₨]\s?[\d,]+(\.\d+)?$",                   # currency
    r"^\d+(\.\d+)?\s?%$",                            # percentage
    r"^(true|false|yes|no|0|1|y|n)$",                # binary
    r"^([A-Z][a-z]+\s?)+$",                          # name-like
    r"^-?\d{1,3}\.\d+\s*,\s*-?\d{1,3}\.\d+$",        # coordinates
    r"^\d+\s+\w+(\s\w+)*$",                          # address-like
]
KEYWORDS = ["street","st","road","rd","avenue","@","http","www",".com",
            "$","%","true","false","yes","no","mr"]
KEYWORDS = KEYWORDS[:15]  # exactly 15

def entropy(s):
    c = Counter(s); n = len(s)
    return -sum(v/n*math.log2(v/n) for v in c.values()) if n else 0.0

def extract(text):
    s = str(text).strip(); n = max(len(s), 1)
    digits = sum(ch.isdigit() for ch in s)
    alpha  = sum(ch.isalpha() for ch in s)
    spaces = sum(ch.isspace() for ch in s)
    special = len(s) - digits - alpha - spaces
    upper = sum(ch.isupper() for ch in s)
    lower = sum(ch.islower() for ch in s)
    length = [len(s), len(s.split()), digits, alpha, spaces, special]          # 6
    pattern = [1 if re.match(p, s, re.I if i in (0,4,8) else 0) else 0
               for i, p in enumerate(PATTERNS)]                                # 12
    stat = [digits/n, alpha/n, upper/(lower+1), special/n, spaces/n,
            len(set(s))/n, entropy(s)]                                         # 7
    low = s.lower()
    content = [1 if k in low else 0 for k in KEYWORDS]                         # 15
    return length + pattern + stat + content

assert len(extract("test@mail.com")) == 40



import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r"F:/NUST_MS_EE_ICS/Unsynced_Stuff/Sem1/AIML/BlueEdge_Project/Datasets/improved_dataset v4.csv")
df["sample"] = df["sample"].fillna("").astype(str)

X = np.array([extract(t) for t in df["sample"]], dtype=np.float32)
le = LabelEncoder()
y = le.fit_transform(df["data_type"])

print(X.shape, y.shape)      # expect (1400, 40) (1400,)
print(list(le.classes_))