# CropWatch: Satellite-Based Crop Monitoring System

Welcome to the CropWatch repository! CropWatch is a satellite-based crop monitoring system designed to provide comprehensive insights into crop health through the analysis of satellite imagery. This project leverages deep learning models, particularly the ResNet50 architecture, to classify biomes and estimate key vegetation indices, such as the Leaf Area Index (LAI) and Chlorophyll content (Cab), from Sentinel-2 satellite data. These metrics are crucial for effective agricultural management, resource allocation, and early detection of crop stress or disease.

## Introduction

CropWatch is designed to empower farmers, agronomists, and researchers by providing actionable insights into crop conditions using satellite imagery. By accurately classifying different biomes and estimating vegetation indices, CropWatch helps monitor crop health, detect potential stress factors, and make informed decisions in agricultural management.

## Features

- **Biome Classification**: Utilize the ResNet50 deep learning model to classify various biomes such as "Forest," "SeaLake," "HerbaceousVegetation," "AnnualCrop," and "Pasture" from satellite imagery, providing a foundational layer for further vegetation analysis.
- **Downlink to Ground Station**: Facilitate the downlink of processed satellite data to ground stations, ensuring that actionable information is readily available for decision-making in the field.

  
- **Vegetation Index Calculation**: Calculate standard vegetation indices such as the Normalized Difference Vegetation Index (NDVI) and Modified Soil-Adjusted Vegetation Index (MSAVI) to assess crop health and productivity.
  
- **LAI and Cab Estimation**: Estimate the Leaf Area Index (LAI) and Chlorophyll content (Cab) using a trained neural network model applied to Sentinel-2 satellite imagery, crucial for monitoring plant health and growth stages.
  
- **Time-Series Analysis**: Track changes in vegetation indices over time to identify trends and detect early signs of crop stress, allowing for timely intervention in agricultural management.
  
- **Heatmap Generation**: Create visual heatmaps that display the spatial distribution of vegetation indices and estimated parameters, helping to quickly identify areas of concern and focus resources effectively.
  
- **Masking and Classification**: Automatically identify and mask areas showing potential signs of crop stress or disease based on changes in LAI and Cab, enabling targeted responses to emerging issues.
  
