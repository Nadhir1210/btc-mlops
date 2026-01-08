#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Azure Deployment Assistant
Guide pas-à-pas pour deployer l'API BTC sur Azure
"""

print("""
╔════════════════════════════════════════════════════════════════╗
║   Azure Deployment Guide - BTC Prediction API                  ║
╚════════════════════════════════════════════════════════════════╝

IMPORTANT POUR AZURE FOR STUDENTS :
────────────────────────────────────

Les abonnements "Azure for Students" ont des restrictions regionales.
Voici les REGIONS AUTORISEES :

  ✓ swedencentral
  ✓ uksouth
  ✓ australiaeast
  ✓ southeastasia

Les regions INTERDITES incluent :
  ✗ westeurope
  ✗ eastus
  ✗ northeurope

════════════════════════════════════════════════════════════════

SOLUTION 1 : Utiliser une region autorisee (swedencentral)
────────────────────────────────────────────────────────────

# Etape 1 : Creer le Resource Group
az group create --name "btc-mlops-rg" --location "swedencentral"

# Etape 2 : Creer Azure Container Registry
az acr create \\
    --resource-group "btc-mlops-rg" \\
    --name "btcmlopsacr2442" \\
    --sku Basic \\
    --admin-enabled true

# Etape 3 : Login a ACR
az acr credential show --name "btcmlopsacr2442" --query "passwords[0].value" -o tsv | \\
    docker login btcmlopsacr2442.azurecr.io -u btcmlopsacr2442 --password-stdin

# Etape 4 : Tag et Push l'image
docker tag btc-prediction-api:latest btcmlopsacr2442.azurecr.io/btc-prediction-api:latest
docker push btcmlopsacr2442.azurecr.io/btc-prediction-api:latest

# Etape 5 : Deployer sur Azure Container Instances
az container create \\
    --resource-group "btc-mlops-rg" \\
    --name "btc-api" \\
    --image "btcmlopsacr2442.azurecr.io/btc-prediction-api:latest" \\
    --registry-login-server "btcmlopsacr2442.azurecr.io" \\
    --registry-username "btcmlopsacr2442" \\
    --registry-password "<PASSWORD_FROM_ACR>" \\
    --dns-name-label "btc-api" \\
    --ports 8000 \\
    --cpu 1 \\
    --memory 2 \\
    --location "swedencentral"

# Etape 6 : Obtenir l'URL publique
az container show \\
    --resource-group "btc-mlops-rg" \\
    --name "btc-api" \\
    --query ipAddress.fqdn \\
    --output tsv

════════════════════════════════════════════════════════════════

SOLUTION 2 : Deploiement Local avec Docker Compose
─────────────────────────────────────────────────────

Pour tester localement avant Azure :

cd D:\\deployement ia\\btc-mlops
docker-compose up -d

URLs locales :
  - API:    http://localhost:8000
  - Swagger: http://localhost:8000/docs
  - MLflow:  http://localhost:5000

════════════════════════════════════════════════════════════════

COUTS ESTIMES (Regions autorisees) :

| Service          | SKU    | Cout/mois |
|─────────────────|────────|──────────|
| ACR             | Basic  | ~5 EUR   |
| Container Inst. | 1-2GB  | ~30 EUR  |
| TOTAL           |        | ~35 EUR  |

════════════════════════════════════════════════════════════════

ALTERNATIVE : Azure App Service

Pour un deploiement plus robuste :

az appservice plan create \\
    --name "btc-mlops-plan" \\
    --resource-group "btc-mlops-rg" \\
    --sku B1 \\
    --is-linux \\
    --location "swedencentral"

az webapp create \\
    --resource-group "btc-mlops-rg" \\
    --plan "btc-mlops-plan" \\
    --name "btc-prediction-api" \\
    --deployment-container-image-name "btcmlopsacr2442.azurecr.io/btc-prediction-api:latest"

════════════════════════════════════════════════════════════════

ETAPES SUIVANTES :

1. Choisir une region AUTORISEE (swedencentral recommande)
2. Creer le Resource Group
3. Creer l'ACR
4. Push l'image Docker
5. Deployer sur ACI ou App Service
6. Configurer le monitoring et les alertes

════════════════════════════════════════════════════════════════
""")

# Regions autorisees pour Azure for Students
regions = {
    "swedencentral": "Recommande - Performance optimale pour Europe",
    "uksouth": "UK - Good performance",
    "australiaeast": "Australie - Zone alternative",
    "southeastasia": "Asie - Bande passante optimale"
}

print("\\nREGIONS AUTORISEES POUR VOTRE ABONNEMENT :")
print("─" * 60)
for region, desc in regions.items():
    print(f"  • {region:20s} - {desc}")
