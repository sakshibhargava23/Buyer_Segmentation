# Machine Learning Based Buyer Segmentation and Investment Profiling
## Real Estate Market Intelligence – Parcl Co. Limited

---

## Abstract

This research applies unsupervised machine learning to segment real estate buyers and profile investment behavior using Parcl client and property transaction data. K-Means and hierarchical clustering reveal four distinct buyer segments—Global Investors, First-Time Buyers, Corporate Buyers, and Luxury Investors—that enable data-driven marketing, financing strategy, and geographic targeting.

---

## 1. Introduction

Real estate markets serve heterogeneous buyer populations: individual homeowners, institutional investors, international buyers, and high-net-worth clients. Treating all buyers uniformly leads to inefficient marketing spend, generic property recommendations, and missed investment opportunities.

This project implements an AI-driven segmentation pipeline for Parcl to discover latent buyer patterns from demographic, behavioral, and transaction data.

---

## 2. Dataset Description

### 2.1 Client Data (2,000 records)

| Feature | Description |
|---------|-------------|
| client_id | Unique client identifier |
| client_type | Individual / Company |
| gender | Buyer gender |
| country | Country of residence |
| region | Geographic region |
| date_of_birth | Age indicator |
| acquisition_purpose | Home / Investment |
| loan_applied | Financing indicator (Yes/No) |
| referral_channel | Customer acquisition source |
| satisfaction_score | Customer satisfaction (1–5) |

### 2.2 Property Data (10,000 records)

Property transactions include sale price, floor area, unit category, tower, listing status, and client reference. Sold properties were aggregated per client to derive investment metrics:

- Number of properties purchased
- Total and average investment
- Average floor area and price per sqft
- Maximum transaction value

---

## 3. Methodology

### Step 1 – Data Cleaning

- Parsed mixed date formats in `date_of_birth`
- Normalized categorical labels (title case, stripped whitespace)
- Removed duplicate client entries
- Imputed missing satisfaction scores with median
- Merged client profiles with aggregated property transactions

### Step 2 – Feature Encoding

| Variable | Encoding |
|----------|----------|
| client_type, region, referral_channel, country | One-Hot Encoding |
| acquisition_purpose, gender | Label Encoding |
| loan_applied | Binary flag |
| age | Derived from date_of_birth |

### Step 3 – Feature Scaling

`StandardScaler` applied to numeric features:

- age, satisfaction_score
- num_properties, total_investment, avg_sale_price
- avg_floor_area, max_sale_price, price_per_sqft
- is_investor, loan_applied_flag

### Step 4 – Clustering Models

**K-Means Clustering**
- Efficient and interpretable partition-based clustering
- Used as primary segmentation model (k=4)

**Hierarchical Clustering (Ward linkage)**
- Validates nested cluster structure
- Dendrogram analysis confirms segment separation

### Step 5 – Optimal Cluster Selection

- **Elbow Method**: Inertia plotted for k=2 to k=10
- **Silhouette Score**: Measures cluster cohesion and separation
- Final model uses k=4 aligned with business segment definitions

### Step 6 – Cluster Interpretation

Each cluster profiled on:

- Investment vs. personal use purpose
- Geographic distribution (country, region)
- Loan/financing behavior
- Demographics (age, client type)
- Transaction volume and investment size

---

## 4. Exploratory Data Analysis (EDA)

### Key Findings

1. **Client mix**: ~90% Individual buyers, ~10% Company buyers
2. **Purpose split**: Roughly balanced between Home and Investment acquisitions
3. **Financing**: Significant portion of buyers apply for loans, especially younger segments
4. **Geography**: USA dominates, with California as the top region; international buyers present in Canada, Germany, and other markets
5. **Investment patterns**: Corporate and high-investment buyers purchase multiple units at higher average prices
6. **Satisfaction**: Luxury and corporate segments show higher satisfaction scores

---

## 5. Segmentation Results

| Cluster | Buyer Type | Characteristics |
|---------|-----------|-----------------|
| C1 | Global Investors | High investment volume, investment-purpose purchases, diverse geography |
| C2 | First-Time Buyers | Younger age profile, high loan dependency, personal use focus |
| C3 | Corporate Buyers | Company client type, multiple property acquisitions |
| C4 | Luxury Investors | High satisfaction, large average transaction values |

---

## 6. Business Recommendations

### Marketing Strategy
- **Global Investors**: Target cross-border investment campaigns and premium portfolio offerings
- **First-Time Buyers**: Emphasize financing options, educational content, and starter properties
- **Corporate Buyers**: B2B sales outreach, bulk purchase incentives, office/commercial units
- **Luxury Investors**: White-glove service, high-end listings, concierge referral programs

### Geographic Targeting
- Concentrate digital spend in top-performing regions (California, Oregon, Quebec)
- Localize messaging by segment density per region

### Financing Products
- Design loan products for First-Time Buyers with high loan application rates
- Offer cash-purchase incentives for Global and Luxury Investor segments

### Product Recommendations
- Match property types to segment preferences (Apartment vs. Office)
- Use segment labels in CRM for personalized property recommendations

---

## 7. Technical Deliverables

| Deliverable | Location |
|-------------|----------|
| Training pipeline | `scripts/train_model.py` |
| Streamlit dashboard | `app/streamlit_app.py` |
| Segmented client data | `models/segmented_clients.csv` |
| Cluster summary | `models/cluster_summary.csv` |
| Evaluation plots | `outputs/figures/` |

---

## 8. Conclusion

AI-driven buyer segmentation transforms Parcl's raw client and transaction data into actionable market intelligence. By identifying four distinct buyer segments through K-Means and hierarchical clustering, Parcl can optimize marketing spend, personalize property recommendations, and improve investor targeting—enabling smarter, data-driven real estate investment decisions.

---

*Unified Mentor | Parcl Co. Limited | Real Estate Market Intelligence Project*
