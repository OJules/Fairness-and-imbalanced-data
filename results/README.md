# 📊 Results Directory - Fairness & Imbalanced Data Analysis

This directory contains evaluation metrics, bias assessments, and performance results from the fairness and imbalanced data experiments.

## 📁 Contents

### **Fairness Evaluation Results**
- `bias_detection_metrics.csv` - Algorithmic bias assessment scores
- `demographic_parity_results.json` - Demographic fairness measurements
- `equalized_odds_analysis.csv` - Equal opportunity metrics
- `fairness_vs_accuracy_tradeoff.png` - Performance-fairness balance visualization
- `protected_group_analysis.csv` - Outcomes by sensitive attributes

### **Imbalanced Data Performance**
- `sampling_techniques_comparison.csv` - SMOTE, ADASYN, undersampling results
- `class_distribution_plots.png` - Before/after resampling visualizations
- `minority_class_performance.json` - Recall, precision for underrepresented classes
- `cost_sensitive_optimization.csv` - Weighted learning results
- `ensemble_methods_evaluation.csv` - Balanced bagging/boosting performance

### **Model Evaluation Metrics**
- `confusion_matrices.png` - Classification performance breakdown by group
- `roc_curves_by_group.png` - ROC analysis for protected attributes
- `precision_recall_curves.png` - PR curves for imbalanced classes
- `calibration_plots.png` - Model calibration across different groups

### **Bias Mitigation Results**
- `pre_processing_impact.csv` - Data transformation effects on fairness
- `in_processing_results.json` - Fairness-constrained model performance
- `post_processing_calibration.csv` - Output adjustment effectiveness
- `bias_reduction_summary.txt` - Overall bias mitigation achievements

### **Statistical Analysis**
- `disparate_impact_ratios.csv` - 80% rule compliance measurements
- `statistical_parity_tests.json` - Group fairness statistical tests
- `individual_fairness_metrics.csv` - Similar individual treatment analysis
- `counterfactual_analysis.png` - What-if scenario results

## 🎯 How to Generate Results

Run the analysis scripts to populate this directory:
```bash
python bias_detection_application.py
python imbalanced_data_detection.py
```

The evaluation workflow:
1. **Baseline Assessment** - Original model bias and performance
2. **Bias Detection** - Systematic fairness evaluation
3. **Mitigation Application** - Fairness improvement techniques
4. **Impact Evaluation** - Final fairness and performance metrics

## 📈 Key Performance Indicators

Expected outputs include:
- **Demographic Parity Ratio** - Group outcome equality (target: 0.8-1.2)
- **Equalized Odds Difference** - Equal opportunity across groups (target: <0.1)
- **Balanced Accuracy** - Performance accounting for class imbalance
- **Fairness-Accuracy Trade-off** - Optimal balance point identification

---

*Note: Results demonstrate the balance between model performance and ethical AI principles.*
