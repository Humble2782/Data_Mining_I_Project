# EDA for Columns 15-28: Setup & Execution Guide

**Analyst:** Artur Loreit  
**Date:** October 29, 2025  
**Assignment:** Exploratory Data Analysis for columns 15-28 of `final_table_proposal.csv`

---

## 📋 Your Assigned Columns

You're responsible for analyzing **14 columns** (indices 15-28):

### Time Features (2 columns)
- `hour` (15) - Hour of day when accident occurred
- `minute` (16) - Minute when accident occurred

### Road/Location Characteristics (12 columns)
- `road_category` (17) - Type of road (motorway, national, etc.)
- `road_number` (18) - Road identification number
- `road_number_index` (19) - Road number index
- `road_number_letter` (20) - Road number letter designation
- `traffic_regime` (21) - Traffic flow pattern
- `number_of_traffic_lanes` (22) - Total lanes on the road
- `reserved_lane_present` (23) - Presence of bike/bus lanes
- `longitudinal_profile` (24) - Road slope (flat, hill, etc.)
- `nearest_reference_marker` (25) - Reference point marker
- `nearest_reference_marker_distance` (26) - Distance to marker
- `horizontal_alignment` (27) - Road curvature
- `width_central_reservation` (28) - Width of median strip

---

## 🚀 Prerequisites & Setup

### Step 1: Ensure Python Environment is Configured

Your project uses the standard Python data science stack. Make sure you have these packages installed:

```powershell
# Navigate to project directory
cd "c:\Users\artur\ALoreit\UNI\Master\Semester 1\Data Mining\Data_Mining_I_Project"

# Install required packages (if not already installed)
pip install pandas numpy matplotlib seaborn scipy jupyter
```

### Step 2: Verify Data File Exists

The EDA notebook expects the data at:
```
data/final_table_proposal.csv
```

Check it exists:
```powershell
Test-Path "data\final_table_proposal.csv"
# Should return: True
```

### Step 3: Open the Notebook

Launch Jupyter and open your EDA notebook:

```powershell
# Start Jupyter Notebook
jupyter notebook

# Then navigate to:
# exploration/Artur/EDA_columns_15_28.ipynb
```

**OR** if you prefer VS Code:
- Simply open `exploration/Artur/EDA_columns_15_28.ipynb` in VS Code
- Select the Python kernel when prompted
- Run cells interactively

---

## 📊 What the Analysis Covers

Your EDA notebook includes:

### 1. **Data Quality Assessment** ✓
- Missing value analysis
- Data type verification
- Unique value counts
- Visual missing data patterns

### 2. **Univariate Analysis** ✓
- Distribution of each feature
- Summary statistics
- Visualizations (histograms, bar charts, box plots)
- Categorical feature breakdowns with proper labels

### 3. **Bivariate Analysis** ✓
- Relationship between YOUR features and `injury_severity` (target)
- Time-based patterns
- Road characteristics impact
- Cross-tabulations and conditional distributions

### 4. **Statistical Testing** ✓
- **Chi-square tests** for categorical features vs injury severity
- **ANOVA** for numerical features vs injury severity
- Significance testing (α = 0.05)

### 5. **Correlation Analysis** ✓
- Correlation matrix for numerical features
- Feature interactions

### 6. **Documentation** ✓
- Template for recording key insights
- Recommendations for modeling phase

---

## 🎯 Expected Deliverables

After running the analysis, you should be able to answer:

1. **Which of your columns have data quality issues?**
   - Missing values > 5%?
   - Unexpected values or outliers?

2. **Which features show strong relationships with injury severity?**
   - Statistically significant (p < 0.05)?
   - Practical significance (meaningful differences)?

3. **Are there temporal patterns in accidents?**
   - Rush hour effects?
   - Night vs day differences?

4. **How do road characteristics affect outcomes?**
   - Motorways vs local roads?
   - Straight sections vs curves?
   - Number of lanes impact?

5. **What recommendations do you have for the modeling phase?**
   - Features to keep/drop?
   - Features needing transformation?
   - Potential feature engineering opportunities?

---

## 💡 Tips for Success

### Before Running Analysis:
- [ ] Read the `column_description.tex` document to understand what each feature means
- [ ] Review the project outline to understand the overall goal
- [ ] Check the data shape to ensure you have the full dataset

### While Running:
- [ ] Run cells sequentially (don't skip)
- [ ] Read the output carefully and take notes
- [ ] Save interesting plots as images if needed
- [ ] Document surprising findings immediately

### After Analysis:
- [ ] Fill in the "Key Insights & Recommendations" section
- [ ] Share interesting findings with your team
- [ ] Identify features that might need special handling in preprocessing
- [ ] Compare your findings with teammates' EDA on other column ranges

---

## 📁 File Structure

```
exploration/Artur/
├── README.md                  # This file
└── EDA_columns_15_28.ipynb   # Your analysis notebook
```

---

## 🔗 Related Files

- **Data:** `../../data/final_table_proposal.csv`
- **Documentation:** `../../documents/column_description.tex`
- **Project Outline:** `../../documents/project_outline.tex`
- **Teammate EDA:**
  - David: `../David/` 
  - Gabriel: `../Gabriel/`

---

## 🆘 Troubleshooting

**Problem:** Can't import packages
```powershell
pip install --upgrade pandas numpy matplotlib seaborn scipy
```

**Problem:** Can't find data file
- Check you're in the right directory
- Verify path with: `Get-ChildItem data\final_table_proposal.csv`

**Problem:** Notebook kernel crashes
- Try restarting the kernel
- Check if dataset is too large (might need to sample)

**Problem:** Plots not showing
- Make sure `%matplotlib inline` is run
- Try `plt.show()` after each plot

---

## ✅ Next Steps After EDA

1. **Document findings** in the insights section
2. **Share with team** - compare across different column ranges
3. **Identify preprocessing needs** for your columns
4. **Prepare recommendations** for feature engineering
5. **Contribute to team discussion** on feature selection

---

**Good luck with your analysis! 📊🚀**

*Last updated: October 29, 2025*
