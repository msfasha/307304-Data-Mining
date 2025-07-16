# Association Rules Mining & Apriori Algorithm
## CS 445: Data Mining

---

## Slide 1: Learning Objectives

### By the end of this lecture, you will:
- Understand association rules and their real-world applications
- Master key metrics: support, confidence, and lift
- See how each metric helps filter and rank rules
- Explain the Apriori algorithm step-by-step
- Apply the algorithm to find actionable business insights

---

## Slide 2: What Are We Actually Associating?

### Real-World Association Examples

**Retail/E-commerce:** 
- **Items ↔ Items:** "Customers who buy diapers also buy beer"
- **Used for:** Product placement, cross-selling, bundle offers

**Healthcare:**
- **Symptoms ↔ Diseases:** "Patients with fever and headache often have flu"
- **Drugs ↔ Side Effects:** "Patients taking Drug A often experience nausea"
- **Used for:** Diagnosis support, drug safety monitoring

**Web Analytics:**
- **Pages ↔ Pages:** "Users who visit Product Page A also visit Product Page B"
- **Actions ↔ Actions:** "Users who click 'Add to Cart' also click 'View Reviews'"
- **Used for:** Website optimization, content recommendations

**The Key Question:** 
**"When X happens, what else is likely to happen?"**

---

## Slide 3: Why Association Rules Matter

### Business Impact Examples

**Supermarket Chain Discovery:**
- Rule: `{Diapers} → {Beer}` (confidence: 65%, lift: 1.8)
- **Action:** Place beer near baby products
- **Result:** 25% increase in beer sales on weekends

**E-commerce Recommendation:**
- Rule: `{iPhone} → {Phone Case}` (confidence: 78%, lift: 2.1)
- **Action:** Auto-suggest phone cases during iPhone checkout
- **Result:** 40% of iPhone buyers also purchase cases

**Website Navigation:**
- Rule: `{View Pricing} → {Contact Sales}` (confidence: 45%, lift: 3.2)
- **Action:** Add prominent "Contact Sales" button on pricing page
- **Result:** 60% increase in sales inquiries

---

## Slide 4: Basic Terminology

### Our Grocery Store Dataset
```
TID | Items Purchased
----|----------------------------------------
T1  | {Bread, Milk, Eggs}
T2  | {Bread, Butter, Cheese}
T3  | {Milk, Butter, Cheese, Yogurt}
T4  | {Bread, Milk, Butter}
T5  | {Bread, Milk, Eggs, Cheese}
T6  | {Butter, Cheese, Yogurt}
T7  | {Bread, Butter, Eggs}
T8  | {Milk, Cheese, Yogurt}
T9  | {Bread, Milk, Butter, Cheese}
T10 | {Eggs, Cheese, Yogurt}
```

### Key Terms
- **Transaction:** One customer's shopping trip
- **Itemset:** Group of items, e.g., {Bread, Milk}
- **Association Rule:** X → Y, e.g., {Bread} → {Milk}

---

## Slide 5: Support - "How Popular Is This Pattern?"

### Definition
Percentage of transactions containing the itemset

### Formula
```
Support(X) = Count(transactions with X) / Total transactions
```

### Calculations from Our Data
```
Support({Bread}) = 6/10 = 60%
Support({Milk}) = 5/10 = 50%
Support({Butter}) = 5/10 = 50%
Support({Cheese}) = 6/10 = 60%
Support({Bread, Milk}) = 3/10 = 30%
Support({Butter, Cheese}) = 4/10 = 40%
```

### Why Support Matters
- **High support:** Common, reliable patterns
- **Low support:** Rare, potentially spurious patterns
- **Threshold:** Typically 5-20% depending on business context

---

## Slide 6: Confidence - "How Reliable Is This Rule?"

### Definition
When X is bought, what's the probability Y is also bought?

### Formula
```
Confidence(X → Y) = Support(X ∪ Y) / Support(X)
```

### Key Comparison - Direction Matters!
```
{Bread} → {Milk}:
Confidence = Support({Bread, Milk}) / Support({Bread})
           = 30% / 60% = 50%

{Milk} → {Bread}:
Confidence = Support({Bread, Milk}) / Support({Milk})
           = 30% / 50% = 60%
```

### Business Insight
**Milk buyers are MORE likely to buy bread (60%) than bread buyers are to buy milk (50%)**
- **Action:** Target bread promotions to milk buyers, not vice versa

---

## Slide 7: Confidence Reveals Asymmetric Relationships

### More Examples from Our Data
```
{Butter} → {Cheese}:
Confidence = Support({Butter, Cheese}) / Support({Butter})
           = 40% / 50% = 80%

{Cheese} → {Butter}:
Confidence = Support({Butter, Cheese}) / Support({Cheese})
           = 40% / 60% = 67%
```

### Business Decision
- **Higher confidence:** `{Butter} → {Cheese}` (80%)
- **Strategy:** When customers buy butter, strongly recommend cheese
- **Expected success rate:** 8 out of 10 butter buyers will accept cheese recommendation

### Why This Matters
**Confidence helps us choose the RIGHT DIRECTION for recommendations**

---

## Slide 8: The Problem with Confidence Alone

### Misleading High Confidence Example
```
{Eggs} → {Cheese}:
Support({Eggs, Cheese}) = 2/10 = 20%
Support({Eggs}) = 4/10 = 40%
Confidence = 20% / 40% = 50%
```

### Seems reasonable, but...
```
Overall popularity of Cheese = Support({Cheese}) = 60%
```

### The Problem
- **Confidence:** 50% of eggs buyers buy cheese
- **Reality:** 60% of ALL customers buy cheese
- **Truth:** Buying eggs actually DECREASES cheese probability!

### This is where LIFT comes to the rescue!

---

## Slide 9: Lift - "Is This Really Meaningful?"

### Definition
How much more likely is Y when X is present vs. Y's general popularity?

### Formula
```
Lift(X → Y) = Confidence(X → Y) / Support(Y)
```

### The Eggs → Cheese Example Revisited
```
Lift({Eggs} → {Cheese}) = 50% / 60% = 0.83
```

### Interpretation
- **Lift < 1:** Negative correlation (buying eggs REDUCES cheese likelihood)
- **Lift = 1:** Independence (no relationship)
- **Lift > 1:** Positive correlation (buying X INCREASES Y likelihood)

### Business Insight
**Don't recommend cheese to eggs buyers - they're LESS likely to buy it!**

---

## Slide 10: Lift Reveals True Relationships

### Positive Correlation Example
```
{Butter} → {Cheese}:
Confidence = 80%
Lift = 80% / 60% = 1.33
```
**Interpretation:** Butter buyers are 33% MORE likely to buy cheese than average

### Independence Example  
```
{Bread} → {Milk}:
Confidence = 50%
Lift = 50% / 50% = 1.00
```
**Interpretation:** Bread purchase doesn't affect milk purchase probability

### Negative Correlation Example
```
{Yogurt} → {Bread}:
Support({Yogurt, Bread}) = 0/10 = 0%
Support({Yogurt}) = 3/10 = 30%
Confidence = 0% / 30% = 0%
Lift = 0% / 60% = 0.00
```
**Interpretation:** Yogurt buyers NEVER buy bread - strong negative correlation

---

## Slide 11: Why We Need All Three Metrics

### Complete Rule Evaluation
```
Rule                  | Support | Confidence | Lift  | Business Value
----------------------|---------|------------|-------|---------------
{Butter} → {Cheese}   |   40%   |    80%     | 1.33  | HIGH ✓
{Milk} → {Bread}      |   30%   |    60%     | 1.00  | NONE ✗
{Eggs} → {Cheese}     |   20%   |    50%     | 0.83  | NEGATIVE ✗
{Cheese} → {Yogurt}   |   15%   |    25%     | 0.83  | WEAK ✗
```

### Each Metric's Role
- **Support:** Filters out rare patterns (need enough data)
- **Confidence:** Ensures rule reliability (high success rate)
- **Lift:** Confirms true correlation (not just coincidence)

### Best Rule: `{Butter} → {Cheese}`
- **40% support:** Frequent enough to be reliable
- **80% confidence:** 4 out of 5 butter buyers will buy cheese
- **1.33 lift:** 33% more likely than random chance

---

## Slide 12: Apriori Algorithm Overview

### Core Principle: Apriori Property
**"All subsets of a frequent itemset must be frequent"**

### Example
If {Bread, Milk, Butter} is frequent, then:
- {Bread, Milk} must be frequent
- {Bread, Butter} must be frequent  
- {Milk, Butter} must be frequent
- {Bread}, {Milk}, {Butter} must be frequent

### Pruning Power
If {Bread, Milk} is NOT frequent, then:
- {Bread, Milk, Butter} cannot be frequent
- {Bread, Milk, Cheese} cannot be frequent
- Any superset containing {Bread, Milk} can be eliminated

---

## Slide 13: Apriori Algorithm Steps

### Algorithm Workflow
1. **Find L₁:** Count all items, keep frequent ones
2. **Generate candidates:** Combine frequent (k-1)-itemsets
3. **Prune candidates:** Remove those with infrequent subsets
4. **Count support:** Scan database for remaining candidates
5. **Filter:** Keep only frequent k-itemsets
6. **Repeat:** Until no new frequent itemsets found
7. **Generate rules:** From all frequent itemsets

### Parameters for Our Example
- **Minimum Support:** 30% (3 out of 10 transactions)
- **Minimum Confidence:** 60%
- **Minimum Lift:** 1.2

---

## Slide 14: Iteration 1 - Find Frequent Items

### Count Individual Items
```
Item     | Count | Support | Status
---------|-------|---------|--------
Bread    |   6   |   60%   |   ✓
Milk     |   5   |   50%   |   ✓
Butter   |   5   |   50%   |   ✓
Cheese   |   6   |   60%   |   ✓
Eggs     |   4   |   40%   |   ✓
Yogurt   |   3   |   30%   |   ✓
```

### Result: L₁ (Frequent 1-itemsets)
**All items meet 30% minimum support threshold**

L₁ = {{Bread}, {Milk}, {Butter}, {Cheese}, {Eggs}, {Yogurt}}

---

## Slide 15: Iteration 2 - Find Frequent Pairs

### Generate All Possible Pairs
6 items → 15 possible pairs

### Count Support for Each Pair
```
Itemset           | Transactions    | Count | Support | Status
------------------|-----------------|-------|---------|--------
{Bread, Milk}     | T1,T4,T5,T9    |   4   |   40%   |   ✓
{Bread, Butter}   | T2,T4,T7,T9    |   4   |   40%   |   ✓
{Bread, Cheese}   | T2,T5,T9       |   3   |   30%   |   ✓
{Bread, Eggs}     | T1,T5,T7       |   3   |   30%   |   ✓
{Bread, Yogurt}   | None           |   0   |    0%   |   ✗
{Milk, Butter}    | T3,T4,T9       |   3   |   30%   |   ✓
{Milk, Cheese}    | T3,T5,T8,T9    |   4   |   40%   |   ✓
{Milk, Eggs}      | T1,T5          |   2   |   20%   |   ✗
{Milk, Yogurt}    | T3,T8          |   2   |   20%   |   ✗
{Butter, Cheese}  | T2,T3,T6,T9    |   4   |   40%   |   ✓
{Butter, Eggs}    | T7             |   1   |   10%   |   ✗
{Butter, Yogurt}  | T3,T6          |   2   |   20%   |   ✗
{Cheese, Eggs}    | T5,T10         |   2   |   20%   |   ✗
{Cheese, Yogurt}  | T3,T6,T8,T10   |   4   |   40%   |   ✓
{Eggs, Yogurt}    | T10            |   1   |   10%   |   ✗
```

---

## Slide 16: Iteration 2 Results

### L₂ (Frequent 2-itemsets)
```
{Bread, Milk}     - 40% support
{Bread, Butter}   - 40% support  
{Bread, Cheese}   - 30% support
{Bread, Eggs}     - 30% support
{Milk, Butter}    - 30% support
{Milk, Cheese}    - 40% support
{Butter, Cheese}  - 40% support
{Cheese, Yogurt}  - 40% support
```

### Key Observations
- **8 frequent pairs** out of 15 possible
- **{Bread, Yogurt}** has 0% support - strong negative correlation
- **{Butter, Cheese}** has highest support (40%) - good combination

---

## Slide 17: Iteration 3 - Generate 3-itemset Candidates

### Step 1: Candidate Generation (Join Step)
Combine frequent 2-itemsets that share exactly one item:
- {Bread, Milk} + {Bread, Butter} → candidate {Bread, Milk, Butter}
- {Bread, Milk} + {Bread, Cheese} → candidate {Bread, Milk, Cheese}
- {Bread, Butter} + {Bread, Cheese} → candidate {Bread, Butter, Cheese}
- {Milk, Butter} + {Milk, Cheese} → candidate {Milk, Butter, Cheese}

### Step 2: Prune Candidates (Based on Apriori Property)
For each candidate, check if ALL its 2-subsets are frequent:

**{Bread, Milk, Butter}:**
- {Bread, Milk} ✓ (in L₂), {Bread, Butter} ✓ (in L₂), {Milk, Butter} ✓ (in L₂)
- **Keep candidate**

**{Bread, Milk, Cheese}:**
- {Bread, Milk} ✓ (in L₂), {Bread, Cheese} ✓ (in L₂), {Milk, Cheese} ✓ (in L₂)
- **Keep candidate**

### Step 3: Ready for Database Scan
4 candidates survive pruning - now we check which ones actually exist in transactions!

---

## Slide 18: Iteration 3 - Count Actual Support in Database

### Step 4: Database Scan (The Crucial Step!)
Now we check: **Do these 3 items actually appear together in real transactions?**

```
Candidate              | Which Transactions? | Count | Support | Status
-----------------------|--------------------|-------|---------|--------
{Bread, Milk, Butter}  | T4, T9            |   2   |   20%   |   ✗
{Bread, Milk, Cheese}  | T5, T9            |   2   |   20%   |   ✗
{Bread, Butter, Cheese}| T2, T9            |   2   |   20%   |   ✗
{Milk, Butter, Cheese} | T3, T9            |   2   |   20%   |   ✗
```

### Let's Verify One Example:
**{Bread, Milk, Butter} appears in:**
- **T4:** {Bread, Milk, Butter} ✓
- **T9:** {Bread, Milk, Butter, Cheese} ✓
- Count: 2 transactions out of 10 = 20% support

### Result: L₃ = {} (Empty Set)
**No 3-itemsets meet our 30% minimum support threshold**

### Algorithm Terminates
Cannot generate any 4-itemsets since L₃ is empty

---

## Slide 19: Generate Association Rules

### From Each Frequent Itemset, Generate All Possible Rules

### Rules from {Butter, Cheese} (40% support):
```
Rule                    | Confidence | Lift  | Evaluation
------------------------|------------|-------|------------
{Butter} → {Cheese}     |    80%     | 1.33  | EXCELLENT ✓
{Cheese} → {Butter}     |    67%     | 1.33  | GOOD ✓
```

### Rules from {Bread, Milk} (40% support):
```
Rule                    | Confidence | Lift  | Evaluation
------------------------|------------|-------|------------
{Bread} → {Milk}        |    67%     | 1.33  | GOOD ✓
{Milk} → {Bread}        |    80%     | 1.33  | EXCELLENT ✓
```

### Rules from {Cheese, Yogurt} (40% support):
```
Rule                    | Confidence | Lift  | Evaluation
------------------------|------------|-------|------------
{Cheese} → {Yogurt}     |    67%     | 2.22  | EXCELLENT ✓
{Yogurt} → {Cheese}     |   100%     | 1.67  | PERFECT ✓
```

---

## Slide 20: Final Rule Rankings

### Applying Our Thresholds (Confidence ≥ 60%, Lift ≥ 1.2)

```
Rank | Rule                  | Support | Confidence | Lift | Business Action
-----|----------------------|---------|------------|------|------------------
1    | {Yogurt} → {Cheese}  |   40%   |   100%     | 1.67 | Always suggest cheese with yogurt
2    | {Milk} → {Bread}     |   40%   |    80%     | 1.33 | Promote bread to milk buyers  
3    | {Butter} → {Cheese}  |   40%   |    80%     | 1.33 | Bundle butter with cheese
4    | {Cheese} → {Yogurt}  |   40%   |    67%     | 2.22 | Strong yogurt promotion to cheese buyers
5    | {Cheese} → {Butter}  |   40%   |    67%     | 1.33 | Cross-sell butter with cheese
6    | {Bread} → {Milk}     |   40%   |    67%     | 1.33 | Promote milk to bread buyers
```

### Top Business Insights
1. **Perfect reliability:** Every yogurt buyer also buys cheese
2. **Strong asymmetry:** Milk buyers more likely to buy bread than vice versa
3. **Dairy synergy:** Butter and cheese are strongly correlated

---

## Slide 21: Why Each Metric Was Essential

### Support Eliminated Rare Patterns
- **{Eggs, Yogurt}:** Only 10% support → Too rare to be reliable
- **{Bread, Yogurt}:** 0% support → Strong negative correlation discovered

### Confidence Revealed Direction
- **{Butter} → {Cheese}:** 80% confidence
- **{Cheese} → {Butter}:** 67% confidence
- **Action:** Prioritize cheese recommendations to butter buyers

### Lift Identified True Correlations
- **{Cheese} → {Yogurt}:** 67% confidence seems modest
- **But lift = 2.22:** Cheese buyers are 122% more likely to buy yogurt!
- **Insight:** This is actually a very strong relationship

### Combined Power
**Without all three metrics, we would miss the strongest business opportunities and waste resources on weak relationships**

---

## Slide 22: Algorithm Complexity & Limitations

### Computational Challenges
- **Database scans:** Multiple passes through data
- **Candidate generation:** Can grow exponentially
- **Memory usage:** Storing all frequent itemsets

### Performance Factors
```
Factor              | Impact on Performance
--------------------|----------------------
Low support threshold | Exponential candidate growth
Dense datasets      | More frequent itemsets
Many items          | Larger search space
Large transactions  | Expensive database scans
```

### When Apriori Struggles
- **Very low support thresholds** (< 1%)
- **High-dimensional data** (> 1000 items)
- **Dense datasets** (most items frequent)

---

## Slide 23: Alternative Algorithms

### FP-Growth
- **Advantage:** Only 2 database scans
- **Method:** Compact FP-tree data structure
- **Best for:** Dense datasets, low support thresholds

### ECLAT  
- **Advantage:** Vertical data representation
- **Method:** Set intersection operations
- **Best for:** Sparse datasets, many unique items

### Modern Approaches
- **Parallel processing:** Distributed Apriori implementations
- **Approximate algorithms:** Trade accuracy for speed
- **Streaming algorithms:** Handle continuous data flows

---

## Slide 24: Practical Business Applications

### Supermarket Chain Strategy
**Discovered Rules:**
- `{Yogurt} → {Cheese}` (100% confidence, 1.67 lift)
- `{Butter} → {Cheese}` (80% confidence, 1.33 lift)

**Business Actions:**
- Place cheese display near yogurt section
- Create butter-cheese bundle promotions
- Stock cheese heavily when yogurt sales increase

### E-commerce Recommendations
**Discovered Rules:**
- `{Milk} → {Bread}` (80% confidence, 1.33 lift)

**Implementation:**
- "Customers who bought milk also bought bread"
- Automatic cross-sell suggestions
- Email marketing campaigns

### Inventory Management
**Negative Correlations:**
- `{Bread, Yogurt}` (0% support)

**Insight:** Customers segment into different dietary preferences
**Action:** Separate promotional campaigns for different customer segments

---

## Slide 25: Summary & Key Takeaways

### What We Learned
1. **Support** filters rare, unreliable patterns
2. **Confidence** reveals directional strength and asymmetry
3. **Lift** distinguishes real correlation from coincidence
4. **All three metrics together** prevent false insights

### Algorithm Insights
- **Apriori property** enables efficient search space pruning
- **Multiple database scans** are the main performance bottleneck
- **Parameter tuning** dramatically affects both performance and results

### Business Value
- **Actionable insights** for cross-selling and recommendations
- **Customer behavior understanding** beyond simple popularity
- **Data-driven decisions** for product placement and marketing

### Critical Success Factors
- **Domain expertise** to interpret and validate rules
- **Proper preprocessing** to clean and prepare data
- **Continuous monitoring** to ensure rules remain relevant

---

## Slide 26: Next Steps & Advanced Topics

### Immediate Applications
- **Implement basic Apriori** on your own datasets
- **Experiment with thresholds** to understand trade-offs
- **Validate discovered rules** with domain experts

### Advanced Techniques
- **Sequential pattern mining:** Time-ordered associations
- **Multi-level associations:** Category hierarchies
- **Constraint-based mining:** User-specified business rules

### Modern Developments
- **Real-time association mining** for streaming data
- **Privacy-preserving techniques** for sensitive data
- **Deep learning approaches** for complex pattern discovery

### Tools to Explore
- **Python:** mlxtend, apyori libraries
- **R:** arules, arulesViz packages
- **Spark:** MLlib for large-scale mining