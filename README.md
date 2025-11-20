# EduGlobe-Insights

Data-driven market analysis using World Bank EdStats to identify high-potential countries for online education expansion. Explores global education indicators, enrollment trends, and infrastructure readiness to guide strategic market entry decisions for high school and university-level learners.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Business Objective](#business-objective)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---



This project evaluates whether the World Bank's education data can provide sufficient and reliable insights to guide **Academy's international expansion strategy** for online education services.

### Specific Goals

- Identify countries with high potential demand for online education services
- Analyze trends in educational indicators to forecast future opportunities
- Provide data-driven recommendations for market prioritization

### Target Market

High school and university-level learners in international markets

---

## 📊 Business Objective

Conduct an exploratory analysis of the World Bank EdStats datasets to assess their relevance, completeness, and potential to inform strategic expansion decisions for Academy's online education platform.

---

## 📁 Dataset

The analysis uses the **World Bank EdStats database**, a comprehensive collection of global education indicators consisting of five interconnected datasets:

### Core Datasets

1. **EdStatsData**  
   Primary quantitative dataset containing indicator values by country and year

2. **EdStatsSeries**  
   Complete catalog of available indicators with definitions and thematic categories

3. **EdStatsCountry**  
   Country-level metadata including ISO codes, regional classifications, and income groups

4. **EdStatsCountry-Series**  
   Links countries to specific indicators with metadata on sources and units

5. **EdStatsFootNote**  
   Explanatory notes and caveats for specific data points

**Data Source:** [World Bank EdStats](https://datacatalog.worldbank.org/search/dataset/0038480)

### Getting the Data

Due to file size constraints, datasets are not included in this repository. 

**Download instructions:**
1. Visit the [World Bank EdStats portal](https://datacatalog.worldbank.org/search/dataset/0038480)
2. Download the CSV files
3. Place them in a `data/` folder in the project root:
