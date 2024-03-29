# KTH DA233X Machine Learning Master Thesis Repo 

**Author**: Isac Lorentz

**Report title**: Shoppin' in the Rain - An Evaluation of the Usefulness of Weather-Based Features for a ML Ranking Model in the Setting of Children's Clothing Online Retailing

# Report file

[Report in DiVA](https://kth.diva-portal.org/smash/get/diva2:1816013/FULLTEXT01.pdf)

# Languages and Libraries
- **Python**
- **SQL** and **Snowflake**
- **GCP**
- **LigtGBM**: for creating the ML ranking models
- **matplotlib** and **seaborn**: for plotting
- **pandas**: for data manipulation, data cleaning and feature engineering
- **plotly**: used for plotting and creating linear regressions
- **scikit-learn**: used for testing assumptions linear regression and for
splitting datasets
- **SciPy**: used for statistics
- **statsmodels**: used for testing assumptions of linear regression 
# Abstract

Online shopping provides numerous benefits to consumers, but large product catalogs make it difficult for shoppers to know the existence and characteristics of every item for sale. To simplify the decision making process, online retailers use ranking models to try to recommend products that are relevant to each individual user while they browse the shopping platforms. Contextual user data such as user location, time, or local weather conditions can serve as valuable features for ranking models as it enables serving of personalized real-time recommendations. Little research has been published on the usefulness of weather-based features for ranking models in the domain of clothing online retailing, which makes additional research into this topic worthwhile. Using Swedish sales- and customer data from Babyshop, an online retailer of children's fashion, this study examines possible correlations between local weather data and sales by comparing differences in daily weather and differences in daily shares of sold items per clothing category for Stockholm and Göteborg. Historical observational weather data from one location each in Stockholm, Göteborg, and Malmö was then featurized and used along with the customers' postal towns, sales features, and sales trend features to evaluate the ranking relevancy of a Gradient-Boosted Decision Trees Learning-to-Rank LightGBM ranking model with weather features against a LightGBM baseline that omitted the weather features and a naive baseline: a popularity-based ranker. Several possible correlations between clothing categories such as shorts, rainwear, shell jackets, winter wear and the daily weather variables feels-like-temperature, solar energy, ultraviolet light index, wind speed, precipitation, snow, and snow depth were found. Evaluation of the ranking relevancy was done using the mean reciprocal rank and the mean average precision @ 10 on a small dataset consisting only of customer data from the postal towns Stockholm, Göteborg, and Malmö and also on a larger dataset where customers in postal towns from larger geographical areas had their home locations approximated as Stockholm, Göteborg or Malmö. The LightGBM rankers beat the naive baseline in three out of four configurations and the ranker with weather features outperformed the LightGBM baseline by 1.1 to 2.2 percent. The results can potentially help clothing online retailers to create more relevant product recommendations.

