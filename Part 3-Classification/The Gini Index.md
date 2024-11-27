# Construct Decision Tree using the Gini Index
<img src="https://raw.githubusercontent.com/msfasha/307304-Data-Mining/main/images/decision_tree_2.png" alt="Play Tennis Table" width="500"/>

### **Step 1: Calculate the Gini Index for the Root Node**
The target variable is **Decision** with two classes: **Yes** and **No**.

- Total samples = \(14\)
- Class distribution:
  - **Yes**: \(9\)
  - **No**: \(5\)

Gini Impurity for the root node:
$$
Gini_{root} = 1 - \left(\frac{9}{14}\right)^2 - \left(\frac{5}{14}\right)^2
$$
Substituting values:
$$
Gini_{root} = 1 - (0.6429)^2 - (0.3571)^2 = 1 - 0.4132 - 0.1275 = 0.4593
$$

---

### **Step 2: Splitting by "Outlook"**
Outlook is a nominal feature. It can take three unique values for **Outlook**: **Sunny**, **Overcast**, **Rainfall**
<img src="https://raw.githubusercontent.com/msfasha/307304-Data-Mining/main/images/outlook_table.png" alt="Outlook" width="500"/>

#### **Subnode: Sunny**
- Samples: \(5\)
  - **Yes**: \(2\)
  - **No**: \(3\)
- Gini Impurity:
$$
Gini_{Sunny} = 1 - \left(\frac{2}{5}\right)^2 - \left(\frac{3}{5}\right)^2
$$
$$
Gini_{Sunny} = 1 - 0.16 - 0.36 = 0.48
$$

#### **Subnode: Overcast**
- Samples: \(4\)
  - **Yes**: \(4\)
  - **No**: \(0\)
- Gini Impurity:
$$
Gini_{Overcast} = 1 - \left(\frac{4}{4}\right)^2 - \left(\frac{0}{4}\right)^2
$$
$$
Gini_{Overcast} = 1 - 1 - 0 = 0
$$

#### **Subnode: Rainfall**
- Samples: \(5\)
  - **Yes**: \(3\)
  - **No**: \(2\)
- Gini Impurity:
$$
Gini_{Rainfall} = 1 - \left(\frac{3}{5}\right)^2 - \left(\frac{2}{5}\right)^2
$$
$$
Gini_{Rainfall} = 1 - 0.36 - 0.16 = 0.48
$$

---

### **Step 3: Weighted Gini Impurity for the Split**
Calculate the weighted Gini impurity after splitting by **Outlook**:

$$
Gini_{Outlook} = \frac{n_{Sunny}}{N} \cdot Gini_{Sunny} + \frac{n_{Overcast}}{N} \cdot Gini_{Overcast} + \frac{n_{Rainfall}}{N} \cdot Gini_{Rainfall}
$$
Where:
- \( n_{Sunny} = 5 \), \( n_{Overcast} = 4 \), \( n_{Rainfall} = 5 \), \( N = 14 \)

Substitute values:
$$
Gini_{Outlook} = \frac{5}{14} \cdot 0.48 + \frac{4}{14} \cdot 0 + \frac{5}{14} \cdot 0.48
$$
$$
Gini_{Outlook} = 0.1714 + 0 + 0.1714 = 0.3428
$$

---

### **Step 4: Gini Gain for "Outlook"**
The reduction in Gini impurity (Gini Gain) is:
$$
Gini\ Gain = Gini_{root} - Gini_{Outlook}
$$
Substitute values:
$$
Gini\ Gain = 0.4593 - 0.3428 = 0.1165
$$
Let’s compute the Gini indices and gains for the next features systematically. We will follow the same steps for **Temperature**, **Humidity**, and **Wind**.

---

### **Step 1: Splitting by "Temperature"**

Unique values for **Temperature**: **Hot**, **Mild**, **Cool**

#### **Subnode: Hot**
- Samples: \(4\)
  - **Yes**: \(2\)
  - **No**: \(2\)
- Gini Impurity:
$$
Gini_{Hot} = 1 - \left(\frac{2}{4}\right)^2 - \left(\frac{2}{4}\right)^2
$$
$$
Gini_{Hot} = 1 - 0.25 - 0.25 = 0.5
$$

#### **Subnode: Mild**
- Samples: \(6\)
  - **Yes**: \(4\)
  - **No**: \(2\)
- Gini Impurity:
$$
Gini_{Mild} = 1 - \left(\frac{4}{6}\right)^2 - \left(\frac{2}{6}\right)^2
$$
$$
Gini_{Mild} = 1 - 0.4444 - 0.1111 = 0.4444
$$

#### **Subnode: Cool**
- Samples: \(4\)
  - **Yes**: \(3\)
  - **No**: \(1\)
- Gini Impurity:
$$
Gini_{Cool} = 1 - \left(\frac{3}{4}\right)^2 - \left(\frac{1}{4}\right)^2
$$
$$
Gini_{Cool} = 1 - 0.5625 - 0.0625 = 0.375
$$

---

### **Step 2: Weighted Gini Impurity for "Temperature"**
$$
Gini_{Temperature} = \frac{n_{Hot}}{N} \cdot Gini_{Hot} + \frac{n_{Mild}}{N} \cdot Gini_{Mild} + \frac{n_{Cool}}{N} \cdot Gini_{Cool}
$$
Where:
- \( n_{Hot} = 4 \), \( n_{Mild} = 6 \), \( n_{Cool} = 4 \), \( N = 14 \)

Substitute values:
$$
Gini_{Temperature} = \frac{4}{14} \cdot 0.5 + \frac{6}{14} \cdot 0.4444 + \frac{4}{14} \cdot 0.375
$$
$$
Gini_{Temperature} = 0.1429 + 0.1905 + 0.1071 = 0.4405
$$

---

### **Step 3: Gini Gain for "Temperature"**
$$
Gini\ Gain = Gini_{root} - Gini_{Temperature}
$$
Substitute values:
$$
Gini\ Gain = 0.4593 - 0.4405 = 0.0188
$$

---

### **Step 4: Splitting by "Humidity"**

Unique values for **Humidity**: **High**, **Normal**

#### **Subnode: High**
- Samples: \(7\)
  - **Yes**: \(3\)
  - **No**: \(4\)
- Gini Impurity:
$$
Gini_{High} = 1 - \left(\frac{3}{7}\right)^2 - \left(\frac{4}{7}\right)^2
$$
$$
Gini_{High} = 1 - 0.1837 - 0.3265 = 0.4898
$$

#### **Subnode: Normal**
- Samples: \(7\)
  - **Yes**: \(6\)
  - **No**: \(1\)
- Gini Impurity:
$$
Gini_{Normal} = 1 - \left(\frac{6}{7}\right)^2 - \left(\frac{1}{7}\right)^2
$$
$$
Gini_{Normal} = 1 - 0.7347 - 0.0204 = 0.2449
$$

---

### **Step 5: Weighted Gini Impurity for "Humidity"**
$$
Gini_{Humidity} = \frac{n_{High}}{N} \cdot Gini_{High} + \frac{n_{Normal}}{N} \cdot Gini_{Normal}
$$
Where:
- \( n_{High} = 7 \), \( n_{Normal} = 7 \), \( N = 14 \)

Substitute values:
$$
Gini_{Humidity} = \frac{7}{14} \cdot 0.4898 + \frac{7}{14} \cdot 0.2449
$$
$$
Gini_{Humidity} = 0.2449 + 0.1225 = 0.3674
$$

---

### **Step 6: Gini Gain for "Humidity"**
$$
Gini\ Gain = Gini_{root} - Gini_{Humidity}
$$
Substitute values:
$$
Gini\ Gain = 0.4593 - 0.3674 = 0.0919
$$

---

### **Step 7: Splitting by "Wind"**

Unique values for **Wind**: **Weak**, **Strong**

#### **Subnode: Weak**
- Samples: \(8\)
  - **Yes**: \(6\)
  - **No**: \(2\)
- Gini Impurity:
$$
Gini_{Weak} = 1 - \left(\frac{6}{8}\right)^2 - \left(\frac{2}{8}\right)^2
$$
$$
Gini_{Weak} = 1 - 0.5625 - 0.0625 = 0.375
$$

#### **Subnode: Strong**
- Samples: \(6\)
  - **Yes**: \(3\)
  - **No**: \(3\)
- Gini Impurity:
$$
Gini_{Strong} = 1 - \left(\frac{3}{6}\right)^2 - \left(\frac{3}{6}\right)^2
$$
$$
Gini_{Strong} = 1 - 0.25 - 0.25 = 0.5
$$

---

### **Step 8: Weighted Gini Impurity for "Wind"**
$$
Gini_{Wind} = \frac{n_{Weak}}{N} \cdot Gini_{Weak} + \frac{n_{Strong}}{N} \cdot Gini_{Strong}
$$
Where:
- \( n_{Weak} = 8 \), \( n_{Strong} = 6 \), \( N = 14 \)

Substitute values:
$$
Gini_{Wind} = \frac{8}{14} \cdot 0.375 + \frac{6}{14} \cdot 0.5
$$
$$
Gini_{Wind} = 0.2143 + 0.2143 = 0.4286
$$

---

### **Step 9: Gini Gain for "Wind"**
$$
Gini\ Gain = Gini_{root} - Gini_{Wind}
$$
Substitute values:
$$
Gini\ Gain = 0.4593 - 0.4286 = 0.0307
$$

---

### **Summary of Gini Gains**
| Feature       | Gini Impurity After Split | Gini Gain |
|---------------|----------------------------|-----------|
| Outlook       | 0.3428                     | 0.1165    |
| Temperature   | 0.4405                     | 0.0188    |
| Humidity      | 0.3674                     | 0.0919    |
| Wind          | 0.4286                     | 0.0307    |

---

### **Best Feature for Split**
- **Outlook** has the highest Gini Gain (\(0.1165\)).
- The first split should be on **Outlook**.

Let me know if you'd like further calculations for the next splits!
