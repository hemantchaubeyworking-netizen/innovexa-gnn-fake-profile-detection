# INNOVEXA Dataset Documentation

## Overview
This dataset contains profile attributes collected from Instagram users to detect fake/automated accounts.

- **Total Profiles:** 5,000
- **Total Features:** 12 input features + 1 target column (`fake`)
- **Class Balance:** Exactly 2,500 Genuine (Class 0, 50.0%) and 2,500 Fake (Class 1, 50.0%)
- **Missing Values:** 0 across all features
- **Duplicate Rows:** 66 duplicate profiles identified and documented

## Schema and Feature Descriptions

| Feature Name | Data Type | Description |
|---|---|---|
| `profile pic` | Binary (0/1) | Whether the account has an avatar/profile image (1 = Yes, 0 = No) |
| `nums/length username` | Float [0, 1] | Ratio of numeric digits to the total length of the username |
| `fullname words` | Integer | Word count of the user's full name field |
| `nums/length fullname` | Float [0, 1] | Ratio of numeric digits to the total length of the full name |
| `name==username` | Binary (0/1) | Whether the full name is identical to the username (1 = Yes, 0 = No) |
| `description length` | Integer | Character length of the profile biography/description |
| `external URL` | Binary (0/1) | Presence of an external link in the bio (1 = Yes, 0 = No) |
| `private` | Binary (0/1) | Privacy setting (1 = Private account, 0 = Public account) |
| `#posts` | Integer | Total number of published media posts |
| `#followers` | Integer | Total follower count |
| `#follows` | Integer | Total following count (accounts followed by the user) |
| `followers_following_ratio` | Float | Ratio of followers to following (`#followers / max(#follows, 1)`) |
| `fake` | Binary (0/1) | **Target Ground Truth**: 0 = Genuine Profile, 1 = Fake Profile |

## Data Preprocessing and Leakage Prevention
1. **Target Exclusion:** The `fake` target column is strictly isolated and excluded from all feature matrices, transformations, and graph construction.
2. **Train / Validation / Test Split:** Stratified 70% Training (3,500 samples), 15% Validation (750 samples), and 15% Test (750 samples) using fixed random seed 42.
3. **Scaler Fitting:** Robust scaling and logarithmic transforms for skewed counts (`#followers`, `#follows`, `#posts`, `followers_following_ratio`) are fitted **strictly on the 70% training split**. No test information leaks into the preprocessor.
4. **Graph Construction Notice:** The dataset does not contain real interaction or follow links. The graph is constructed as a k-Nearest Neighbor (kNN) feature-similarity graph on normalized profile attributes.
